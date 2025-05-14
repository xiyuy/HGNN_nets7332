import os
import yaml
import os.path as osp


def get_config(dir='config/config.yaml'):
    # add direction join function when parse the yaml file
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    # add string concatenation function when parse the yaml file
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)

    with open(dir, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    check_dirs(cfg)

    # Add additional checks for ranking configuration
    if 'task_type' in cfg and cfg['task_type'] in ['node_ranking', 'ranking_comparison', 'feature_comparison']:
        check_dir(cfg['saved_models_folder'])

        # Ensure ranking-specific settings are present
        if 'ranking_mode' not in cfg:
            cfg['ranking_mode'] = 'mse'
        if 'use_edge_dependent' not in cfg:
            cfg['use_edge_dependent'] = True
        if 'compare_with_baseline' not in cfg:
            cfg['compare_with_baseline'] = True
        if 'save_rankings' not in cfg:
            cfg['save_rankings'] = True

        # Add feature comparison settings if not present
        if 'use_features' not in cfg:
            cfg['use_features'] = True
        if 'compare_with_featureless' not in cfg:
            cfg['compare_with_featureless'] = True

    return cfg


def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')


def check_dirs(cfg):
    check_dir(cfg['data_root'], mk_dir=False)
    check_dir(cfg['result_root'])
    check_dir(cfg['ckpt_folder'])
    check_dir(cfg['result_sub_folder'])

    # Create saved_models folder if specified in config
    if 'saved_models_folder' in cfg:
        check_dir(cfg['saved_models_folder'])
