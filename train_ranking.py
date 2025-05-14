import os
import time
import copy
import torch
import torch.optim as optim
import numpy as np
import random
import utils.new_utils as hgut
from models.HGNN_ranking import HGNN_Ranking, HGNN_Ranking_Base
from config import get_config
from datasets.data_helper_ranking import load_AMiner_construct_H_and_R
from scipy.stats import kendalltau, spearmanr
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')


def train_ranking_model(model, optimizer, scheduler, fts, G, ground_truth=None,
                        num_epochs=25, print_freq=100, mode='mse', seed=42):
    """Train a ranking model with optional ground truth."""
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = -float('inf') if ground_truth is not None else float('inf')

    # Prepare ground truth based on mode
    if mode == 'mse' and ground_truth is not None:
        gt_np = ground_truth.squeeze()
        min_val, max_val = gt_np.min(), gt_np.max()
        normalized_gt = (gt_np - min_val) / (max_val -
                                             min_val) if max_val > min_val else np.zeros_like(gt_np)
        ground_truth_tensor = torch.tensor(
            normalized_gt, device=G.device).float()
        criterion = torch.nn.MSELoss()
        print(
            f"Using normalized ground truth. Range: [{min_val:.4f}, {max_val:.4f}]")
    elif mode == 'pairwise' and ground_truth is not None:
        ground_truth_tensor = torch.tensor(
            ground_truth.squeeze(), device=G.device).float()

        def pairwise_ranking_loss(predictions, targets):
            n = predictions.size(0)
            pairs_i, pairs_j = torch.triu_indices(n, n, offset=1)
            target_diffs = torch.sign(targets[pairs_i] - targets[pairs_j])
            pred_diffs = predictions[pairs_i] - predictions[pairs_j]
            return torch.mean(torch.log(1 + torch.exp(-target_diffs * pred_diffs)))
        criterion = pairwise_ranking_loss
        print("Using pairwise ranking loss")
    else:
        criterion = torch.nn.MSELoss()
        ground_truth_tensor = torch.linspace(0, 1, G.shape[0], device=G.device)
        print("Using diversity-encouraging target")

    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        # Forward pass (handle both model types)
        outputs = model(fts, G) if isinstance(
            model, HGNN_Ranking) else model(G)
        loss = criterion(outputs, ground_truth_tensor)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation and progress reporting
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}/{num_epochs-1}, Loss: {loss.item():.4f}')

            # Evaluate using correlation metrics if ground truth available
            model.eval()
            with torch.no_grad():
                eval_out = model(fts, G) if isinstance(
                    model, HGNN_Ranking) else model(G)

                if ground_truth is not None:
                    preds = eval_out.cpu().detach().numpy()
                    gt = ground_truth.squeeze()
                    tau, _ = kendalltau(preds, gt)
                    rho, _ = spearmanr(preds, gt)
                    print(f'Kendall Tau: {tau:.4f}, Spearman Rho: {rho:.4f}')

                    # Save best model based on correlation
                    if tau > best_metric:
                        best_metric = tau
                        best_model_wts = copy.deepcopy(model.state_dict())
                        print(f'New best model: Tau = {tau:.4f}')
                else:
                    # Save best model based on loss
                    if loss.item() < best_metric:
                        best_metric = loss.item()
                        best_model_wts = copy.deepcopy(model.state_dict())
            print('-' * 50)  # Cleaner separator line

    # Load best model and report training time
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    if ground_truth is not None:
        print(f'Best Kendall Tau: {best_metric:.4f}')
    else:
        print(f'Best loss: {best_metric:.4f}')

    return model


def generate_rankings(model, fts=None, G=None):
    """Generate global rankings for all nodes."""
    model.eval()
    with torch.no_grad():
        scores = model(fts, G).cpu().numpy() if isinstance(
            model, HGNN_Ranking) else model(G).cpu().numpy()
        ranking_indices = np.argsort(-scores)  # Higher score = better rank
        ranking_scores = scores[ranking_indices]
        node_to_rank = {node_idx: rank for rank,
                        node_idx in enumerate(ranking_indices)}
        return ranking_indices, ranking_scores, node_to_rank


def evaluate_rankings(predictions, ground_truth):
    """Evaluate rankings against ground truth."""
    if ground_truth is None:
        return None, None, None, None
    tau, p_tau = kendalltau(predictions, ground_truth)
    rho, p_rho = spearmanr(predictions, ground_truth)
    return tau, p_tau, rho, p_rho


def compare_rankings(model1_ranking, model2_ranking, model1_scores=None, model2_scores=None,
                     ground_truth=None, model1_name="Model 1", model2_name="Model 2",
                     save_output=True, output_dir='results/rankings'):
    """Compare rankings between two models."""
    # Calculate correlation between the two rankings
    tau, p_value = kendalltau(np.argsort(
        model1_ranking), np.argsort(model2_ranking))
    print(
        f'Correlation between {model1_name} and {model2_name}: Kendall Tau = {tau:.4f}, p = {p_value:.4f}')

    # Print top-ranked nodes
    for name, rankings, scores in [
        (model1_name, model1_ranking, model1_scores),
        (model2_name, model2_ranking, model2_scores)
    ]:
        print(f'\nTop 10 nodes using {name}:')
        for i in range(min(10, len(rankings))):
            node_idx = rankings[i]
            score_str = f", Score: {scores[i]:.4f}" if scores is not None else ""
            print(f'  Rank {i+1}: Node {node_idx}{score_str}')

    # Evaluate against ground truth if available
    if ground_truth is not None:
        m1_tau, _, m1_rho, _ = evaluate_rankings(
            model1_scores if model1_scores is not None else model1_ranking, ground_truth)
        m2_tau, _, m2_rho, _ = evaluate_rankings(
            model2_scores if model2_scores is not None else model2_ranking, ground_truth)

        print('\nCorrelations with ground truth:')
        print(
            f'  {model1_name}: Kendall Tau = {m1_tau:.4f}, Spearman Rho = {m1_rho:.4f}')
        print(
            f'  {model2_name}: Kendall Tau = {m2_tau:.4f}, Spearman Rho = {m2_rho:.4f}')
        better_model = model1_name if m1_tau > m2_tau else model2_name
        print(f"  {better_model} performs better according to Kendall's Tau")

    # Find nodes with largest rank differences
    differences = []
    for node in range(min(len(model1_ranking), len(model2_ranking))):
        rank_m1 = np.where(model1_ranking == node)[
            0][0] if node in model1_ranking else -1
        rank_m2 = np.where(model2_ranking == node)[
            0][0] if node in model2_ranking else -1
        if rank_m1 >= 0 and rank_m2 >= 0:
            differences.append((node, rank_m2 - rank_m1))

    differences.sort(key=lambda x: abs(x[1]), reverse=True)
    print(
        f'\n10 nodes with largest rank differences ({model1_name} vs {model2_name}):')
    for i in range(min(10, len(differences))):
        node, diff = differences[i]
        rank_m1 = np.where(model1_ranking == node)[0][0]
        rank_m2 = np.where(model2_ranking == node)[0][0]
        print(
            f'  Node {node}: {model1_name} rank {rank_m1}, {model2_name} rank {rank_m2}, Diff: {diff}')

    # Save comparison if requested
    if save_output:
        os.makedirs(output_dir, exist_ok=True)

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'node_id': list(range(min(len(model1_ranking), len(model2_ranking)))),
            f'rank_{model1_name}': [np.where(model1_ranking == node)[0][0] if node in model1_ranking else -1
                                    for node in range(min(len(model1_ranking), len(model2_ranking)))],
            f'rank_{model2_name}': [np.where(model2_ranking == node)[0][0] if node in model2_ranking else -1
                                    for node in range(min(len(model1_ranking), len(model2_ranking)))]
        })

        # Add scores if available
        if model1_scores is not None:
            comparison_df[f'score_{model1_name}'] = [model1_scores[np.where(model1_ranking == node)[0][0]]
                                                     if node in model1_ranking else -1
                                                     for node in range(min(len(model1_ranking), len(model2_ranking)))]
        if model2_scores is not None:
            comparison_df[f'score_{model2_name}'] = [model2_scores[np.where(model2_ranking == node)[0][0]]
                                                     if node in model2_ranking else -1
                                                     for node in range(min(len(model1_ranking), len(model2_ranking)))]

        # Add ground truth if available
        if ground_truth is not None:
            comparison_df['ground_truth'] = [ground_truth[node]
                                             for node in range(min(len(model1_ranking), len(model2_ranking)))]

        comparison_df['rank_difference'] = comparison_df[f'rank_{model2_name}'] - \
            comparison_df[f'rank_{model1_name}']
        comparison_df['abs_diff'] = comparison_df['rank_difference'].abs()
        comparison_df = comparison_df.sort_values('abs_diff', ascending=False)

        # Save to CSV
        filename = f'{model1_name}_vs_{model2_name}_comparison.csv'.replace(
            ' ', '_').lower()
        output_file = os.path.join(output_dir, filename)
        comparison_df.to_csv(output_file, index=False)
        print(f"\nRanking comparison saved to {output_file}")

    return tau, p_value


def run_feature_vs_featureless(data_dir, Pi_version='from_P', save_output=True):
    """Compare rankings between feature-based and feature-less models."""
    # Load data and prepare tensors
    H, R, E_weights, X, Y, idx_train, idx_test = load_AMiner_construct_H_and_R(
        data_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fts = torch.Tensor(X).to(device)

    # Generate G matrix
    G = hgut.generate_G_from_H(H, R, E_weights, Pi_version=Pi_version)
    G_tensor = torch.Tensor(G).to(device)

    # Common training parameters
    train_params = {
        'ground_truth': Y,
        'num_epochs': cfg['max_epoch'],
        'print_freq': cfg['print_freq'],
        'mode': cfg['ranking_mode'],
        'seed': cfg['seed']
    }

    # Create and train feature-based model
    print("\n" + "="*70)
    print("TRAINING FEATURE-BASED MODEL (HGNN_Ranking)")
    print("="*70)
    model_feat = HGNN_Ranking(
        in_ch=fts.shape[1],
        n_hid=cfg['n_hid'],
        dropout=cfg['drop_out']
    ).to(device)

    optimizer_feat = optim.Adam(
        model_feat.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    scheduler_feat = optim.lr_scheduler.MultiStepLR(
        optimizer_feat,
        milestones=cfg['milestones'],
        gamma=cfg['gamma']
    )

    model_feat = train_ranking_model(
        model=model_feat,
        optimizer=optimizer_feat,
        scheduler=scheduler_feat,
        fts=fts,
        G=G_tensor,
        **train_params
    )

    feat_ranking_indices, feat_ranking_scores, _ = generate_rankings(
        model_feat, fts, G_tensor)

    # Create and train feature-less model
    print("\n" + "="*70)
    print("TRAINING FEATURE-LESS MODEL (HGNN_Ranking_Base)")
    print("="*70)
    model_base = HGNN_Ranking_Base(
        n_nodes=G.shape[0],
        n_hid=cfg['n_hid'],
        dropout=cfg['drop_out']
    ).to(device)

    optimizer_base = optim.Adam(
        model_base.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    scheduler_base = optim.lr_scheduler.MultiStepLR(
        optimizer_base,
        milestones=cfg['milestones'],
        gamma=cfg['gamma']
    )

    model_base = train_ranking_model(
        model=model_base,
        optimizer=optimizer_base,
        scheduler=scheduler_base,
        fts=None,
        G=G_tensor,
        **train_params
    )

    base_ranking_indices, base_ranking_scores, _ = generate_rankings(
        model_base, None, G_tensor)

    # Compare rankings
    print("\n" + "="*70)
    print("COMPARING FEATURE-BASED VS FEATURE-LESS RANKINGS")
    print("="*70)
    tau, p_value = compare_rankings(
        feat_ranking_indices, base_ranking_indices,
        feat_ranking_scores, base_ranking_scores,
        ground_truth=Y.squeeze() if Y is not None else None,
        model1_name="Feature-based", model2_name="Feature-less",
        save_output=save_output,
        output_dir=os.path.join(data_dir, 'rankings')
    )

    # Save models if requested
    if save_output:
        os.makedirs(cfg['saved_models_folder'], exist_ok=True)
        for model_name, model in [('feature_based', model_feat), ('feature_less', model_base)]:
            torch.save(model.state_dict(),
                       os.path.join(cfg['saved_models_folder'], f'{model_name}_ranking_model.pth'))
        print(f"Models saved to {cfg['saved_models_folder']}")

    return tau, p_value, feat_ranking_indices, base_ranking_indices


def run_edge_dependent_vs_independent(data_dir, Pi_version='from_P', save_output=True):
    """Compare rankings between edge-dependent and edge-independent vertex weights."""
    # Load data and prepare tensors
    H, R, E_weights, X, Y, idx_train, idx_test = load_AMiner_construct_H_and_R(
        data_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fts = torch.Tensor(X).to(device)

    # Common training parameters
    train_params = {
        'ground_truth': Y,
        'num_epochs': cfg['max_epoch'],
        'print_freq': cfg['print_freq'],
        'mode': cfg['ranking_mode'],
        'seed': cfg['seed']
    }

    # Create and train edge-dependent (HD) model
    print("\n" + "="*70)
    print("TRAINING EDGE-DEPENDENT MODEL (HD)")
    print("="*70)
    G_HD = torch.Tensor(hgut.generate_G_from_H(
        H, R, E_weights, Pi_version=Pi_version)).to(device)
    model_HD = HGNN_Ranking(
        in_ch=fts.shape[1],
        n_hid=cfg['n_hid'],
        dropout=cfg['drop_out']
    ).to(device)

    optimizer_HD = optim.Adam(
        model_HD.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    scheduler_HD = optim.lr_scheduler.MultiStepLR(
        optimizer_HD,
        milestones=cfg['milestones'],
        gamma=cfg['gamma']
    )

    model_HD = train_ranking_model(
        model=model_HD,
        optimizer=optimizer_HD,
        scheduler=scheduler_HD,
        fts=fts,
        G=G_HD,
        **train_params
    )

    hd_ranking_indices, hd_ranking_scores, _ = generate_rankings(
        model_HD, fts, G_HD)

    # Create and train edge-independent (HT) model
    print("\n" + "="*70)
    print("TRAINING EDGE-INDEPENDENT MODEL (HT)")
    print("="*70)
    R_HT = R.copy()
    R_HT[R_HT != 0] = 1
    G_HT = torch.Tensor(hgut.generate_G_from_H(
        H, R_HT, E_weights, Pi_version=Pi_version)).to(device)
    model_HT = HGNN_Ranking(
        in_ch=fts.shape[1],
        n_hid=cfg['n_hid'],
        dropout=cfg['drop_out']
    ).to(device)

    optimizer_HT = optim.Adam(
        model_HT.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    scheduler_HT = optim.lr_scheduler.MultiStepLR(
        optimizer_HT,
        milestones=cfg['milestones'],
        gamma=cfg['gamma']
    )

    model_HT = train_ranking_model(
        model=model_HT,
        optimizer=optimizer_HT,
        scheduler=scheduler_HT,
        fts=fts,
        G=G_HT,
        **train_params
    )

    ht_ranking_indices, ht_ranking_scores, _ = generate_rankings(
        model_HT, fts, G_HT)

    # Compare rankings
    print("\n" + "="*70)
    print("COMPARING EDGE-DEPENDENT VS EDGE-INDEPENDENT RANKINGS")
    print("="*70)
    tau, p_value = compare_rankings(
        hd_ranking_indices, ht_ranking_indices,
        hd_ranking_scores, ht_ranking_scores,
        ground_truth=Y.squeeze() if Y is not None else None,
        model1_name="Edge-dependent", model2_name="Edge-independent",
        save_output=save_output,
        output_dir=os.path.join(data_dir, 'rankings')
    )

    # Save models if requested
    if save_output:
        os.makedirs(cfg['saved_models_folder'], exist_ok=True)
        for model_name, model in [('hd', model_HD), ('ht', model_HT)]:
            torch.save(model.state_dict(),
                       os.path.join(cfg['saved_models_folder'], f'{model_name}_ranking_model.pth'))
        print(f"Models saved to {cfg['saved_models_folder']}")

    return tau, p_value, hd_ranking_indices, ht_ranking_indices


def run_node_ranking(data_dir, Pi_version='from_P', save_output=True):
    """Run standard node ranking with a single model."""
    # Load data and prepare tensors
    H, R, E_weights, X, Y, idx_train, idx_test = load_AMiner_construct_H_and_R(
        data_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fts = torch.Tensor(X).to(device)

    # Generate G matrix
    G = hgut.generate_G_from_H(H, R, E_weights, Pi_version=Pi_version)
    G_tensor = torch.Tensor(G).to(device)

    # Create and train model
    print("\n" + "="*70)
    print("TRAINING NODE RANKING MODEL")
    print("="*70)
    model = HGNN_Ranking(
        in_ch=fts.shape[1],
        n_hid=cfg['n_hid'],
        dropout=cfg['drop_out']
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg['milestones'],
        gamma=cfg['gamma']
    )

    model = train_ranking_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        fts=fts,
        G=G_tensor,
        ground_truth=Y,
        num_epochs=cfg['max_epoch'],
        print_freq=cfg['print_freq'],
        mode=cfg['ranking_mode'],
        seed=cfg['seed']
    )

    # Generate and evaluate rankings
    ranking_indices, ranking_scores, _ = generate_rankings(
        model, fts, G_tensor)

    # Display top ranked nodes with H-Index
    print('\nTop 15 ranked nodes:')
    for i in range(min(15, len(ranking_indices))):
        node_idx = ranking_indices[i]
        print(f'  Rank {i+1}: Node {node_idx}, Score: {ranking_scores[i]:.4f}, ' +
              f'H-Index: {Y[node_idx][0] if Y is not None else "Unknown"}')

    # Evaluate against ground truth
    if Y is not None:
        tau, _, rho, _ = evaluate_rankings(ranking_scores, Y.squeeze())
        print(f'\nCorrelation with ground truth (H-Index):')
        print(f'  Kendall Tau: {tau:.4f}, Spearman Rho: {rho:.4f}')

    # Save results if requested
    if save_output:
        os.makedirs(cfg['saved_models_folder'], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(
            cfg['saved_models_folder'], 'ranking_model.pth'))

        # Save rankings to CSV
        ranking_df = pd.DataFrame({
            'node_id': ranking_indices,
            'rank': np.arange(len(ranking_indices)),
            'score': ranking_scores
        })

        if Y is not None:
            ranking_df['h_index'] = [Y[node_idx][0]
                                     for node_idx in ranking_indices]

        output_file = os.path.join(
            cfg['saved_models_folder'], 'rankings_with_h_index.csv')
        ranking_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return model, ranking_indices, ranking_scores


def main():
    # Set seeds for reproducibility
    for seed_func in [random.seed, np.random.seed, torch.manual_seed]:
        seed_func(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])

    print("\n" + "="*70)
    print(f"STARTING EXPERIMENT: {cfg['task_type']}")
    print(f"Pi matrix version: {cfg['Pi_version']}")
    print("="*70)

    # Run the appropriate experiment based on task type
    if cfg['task_type'] == 'node_ranking':
        run_node_ranking(
            cfg['data_root'],
            Pi_version=cfg['Pi_version'],
            save_output=cfg['save_rankings']
        )
    elif cfg['task_type'] == 'ranking_comparison':
        run_edge_dependent_vs_independent(
            cfg['data_root'],
            Pi_version=cfg['Pi_version'],
            save_output=cfg['save_rankings']
        )
    elif cfg['task_type'] == 'feature_comparison':
        run_feature_vs_featureless(
            cfg['data_root'],
            Pi_version=cfg['Pi_version'],
            save_output=cfg['save_rankings']
        )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == '__main__':
    main()
