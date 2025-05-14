import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_AMiner_construct_H_and_R(inputdir, test_ratio=0.2, seed=123):
    """
    Load features and construct hypergraph incidence matrix H and 
    edge-dependent vertex weights matrix R from AMiner data.

    Parameters:
    -----------
    inputdir : str
        Directory path containing the AMiner dataset files
    test_ratio : float, optional
        Ratio of test data, by default 0.2
    seed : int, optional
        Random seed for reproducibility, by default 123

    Returns:
    --------
    H : numpy.ndarray
        Hypergraph incidence matrix (V x E)
    R : numpy.ndarray
        Edge-dependent vertex weights matrix (E x V)
    E_weights : numpy.ndarray
        Hyperedge weights (1 x E)
    X : numpy.ndarray
        Node feature matrix (V x F)
    Y : numpy.ndarray
        Node HIndex values as ground truth rankings (V x 1)
    idx_train : numpy.ndarray
        Indices of training nodes
    idx_test : numpy.ndarray
        Indices of test nodes
    """
    print("Loading AMiner data...")

    # Function to normalize author IDs for matching
    def normalize_author_id(author_id):
        return ''.join(c for c in str(author_id) if c.isalnum()).lower()

    # Initialize dictionaries
    sampled_paperid = set()
    papers = []  # (author, pos)
    paperid2hindex = {}
    hindex2paperid = {}
    numhedges = 0
    authorid2vindex = {}
    vindex2authorid = {}
    numnodes = 0
    hindex2cite = {}
    vindex2hindex = {}   # Store HIndex for each author node

    # Load sampled paper IDs
    with open(os.path.join(inputdir, "sampled_paperid_10000.txt"), "r") as f:
        for line in f.readlines():
            paperid = line.rstrip().split("\t")[0]
            sampled_paperid.add(paperid)

    # Load hypergraph structure (papers and authors)
    with open(os.path.join(inputdir, "hypergraph.txt"), "r") as f, \
            open(os.path.join(inputdir, "hypergraph_pos.txt"), "r") as pf:
        for line, pline in zip(f.readlines(), pf.readlines()):
            nodes = line.rstrip().split("\t")
            node_poses = pline.rstrip().split("\t")[1:]
            paperid = nodes[0]
            nodes = nodes[1:]

            if len(nodes) == 1 or paperid not in sampled_paperid:
                continue

            hindex = numhedges
            paperid2hindex[paperid] = hindex
            hindex2paperid[hindex] = paperid
            numhedges += 1

            paper = []
            poses = []
            for authorid, _vpos in zip(nodes, node_poses):
                if authorid not in authorid2vindex:
                    vindex = numnodes
                    authorid2vindex[authorid] = vindex
                    vindex2authorid[vindex] = authorid
                    numnodes += 1
                vindex = authorid2vindex[authorid]

                # Set position weight: 2 for first/last author, 1 for middle authors
                vpos = 2 if int(_vpos) == 1 or int(_vpos) == len(nodes) else 1

                paper.append(int(vindex))
                poses.append(int(vpos))
            papers.append((paper, poses))

    # Create mapping between normalized and original author IDs
    author_id_mapping = {}
    for author_id in authorid2vindex.keys():
        normalized_id = normalize_author_id(author_id)
        author_id_mapping[normalized_id] = author_id

    # Load paper citations if available
    try:
        with open(os.path.join(inputdir, "hypergraph_citation.txt"), "r") as f:
            for line in f.readlines():
                paperid, citation = line.rstrip().split("\t")
                if paperid in paperid2hindex:
                    hindex = paperid2hindex[paperid]
                    hindex2cite[hindex] = int(citation)
    except FileNotFoundError:
        print("Warning: hypergraph_citation.txt not found")
        # Default citation value if file not found
        for hindex in range(numhedges):
            hindex2cite[hindex] = 1

    # Load author features and HIndex from author_info.txt
    author_info_file = os.path.join(inputdir, "author_info.txt")
    author_features = {}  # Dictionary to store features for each author

    if os.path.exists(author_info_file):
        print("Loading author features and HIndex from author_info.txt...")
        with open(author_info_file, "r") as f:
            # Skip header line
            header = f.readline().strip().split(',')

            # Find column indices
            try:
                paper_count_idx = header.index("PaperCount")
                citation_idx = header.index("Citation")
                hindex_idx = header.index("HIndex")
            except ValueError:
                print("Warning: Could not find all required columns in author_info.txt")
                paper_count_idx, citation_idx, hindex_idx = 1, 2, 3  # Default indices

            # Process each author
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) > max(paper_count_idx, citation_idx, hindex_idx):
                    raw_author_id = parts[0]
                    normalized_id = normalize_author_id(raw_author_id)

                    # Check if this author exists in our hypergraph
                    if normalized_id in author_id_mapping:
                        original_author_id = author_id_mapping[normalized_id]
                        vindex = authorid2vindex[original_author_id]

                        try:
                            # Extract HIndex
                            hindex_value = float(
                                parts[hindex_idx]) if hindex_idx < len(parts) else 0.0
                            vindex2hindex[vindex] = hindex_value

                            # Extract features
                            paper_count = float(parts[paper_count_idx])
                            citation = float(parts[citation_idx])
                            author_features[original_author_id] = [
                                paper_count, citation]
                        except (ValueError, IndexError):
                            # Skip lines with invalid data
                            continue
    else:
        print("Warning: author_info.txt not found")

    # Stats on how many authors have HIndex and features
    hindex_count = len(vindex2hindex)
    feature_count = len(author_features)
    print(
        f"Got HIndex for {hindex_count} out of {numnodes} authors ({hindex_count/numnodes*100:.1f}%)")
    print(
        f"Got features for {feature_count} out of {numnodes} authors ({feature_count/numnodes*100:.1f}%)")

    # Fill in missing HIndex values
    for vindex in range(numnodes):
        if vindex not in vindex2hindex:
            vindex2hindex[vindex] = 0.0  # Default to 0 for missing HIndex

    # Create feature matrix X
    X = np.zeros((numnodes, 2))  # 2 features: PaperCount and Citation

    # Calculate median values for missing authors
    if author_features:
        features_array = np.array(list(author_features.values()))
        median_paper_count = np.median(features_array[:, 0])
        median_citation = np.median(features_array[:, 1])
    else:
        median_paper_count, median_citation = 1.0, 0.0

    # Fill in features for each node
    for author_id, vindex in authorid2vindex.items():
        if author_id in author_features:
            X[vindex] = author_features[author_id]
        else:
            X[vindex] = [median_paper_count, median_citation]

    # Apply log transformation to citation counts and standardize features
    X[:, 1] = np.log1p(X[:, 1])
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    feature_stds[feature_stds == 0] = 1.0
    X = (X - feature_means) / feature_stds

    # Construct hypergraph incidence matrix H (V x E)
    H = np.zeros((numnodes, numhedges))
    for i, (paper, _) in enumerate(papers):
        for node in paper:
            H[node, i] = 1

    # Construct edge-dependent vertex weights matrix R (E x V)
    R = np.zeros((numhedges, numnodes))
    for i, (paper, poses) in enumerate(papers):
        for node, pos in zip(paper, poses):
            R[i, node] = pos

    # Construct hyperedge weights
    E_weights = np.ones(numhedges)
    for hindex, citation in hindex2cite.items():
        E_weights[hindex] = np.log(citation + 1)

    # Create Y as HIndex values
    Y = np.zeros((numnodes, 1))
    for vindex, hindex_value in vindex2hindex.items():
        Y[vindex] = hindex_value

    # Split indices for training and testing
    all_indices = np.arange(numnodes)

    # Use quantile-based stratification for the ranking task
    Y_quantiles = np.zeros_like(Y, dtype=int)
    n_bins = 4
    for i in range(n_bins):
        quantile = np.quantile(Y, (i + 1) / n_bins)
        Y_quantiles[Y <= quantile] = i

    idx_train, idx_test = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=Y_quantiles
    )

    print(
        f"Loaded hypergraph with {numnodes} nodes and {numhedges} hyperedges")
    print(f"H shape: {H.shape}, R shape: {R.shape}")
    print(
        f"HIndex values range: min={np.min(Y)}, max={np.max(Y)}, mean={np.mean(Y)}")
    print(f"Training samples: {len(idx_train)}, Test samples: {len(idx_test)}")

    return H, R, E_weights, X, Y, idx_train, idx_test
