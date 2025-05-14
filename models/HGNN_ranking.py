import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNN_Ranking(nn.Module):
    """
    Feature-based hypergraph neural network for ranking
    Uses node features to predict ranking scores
    """

    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_Ranking, self).__init__()
        self.dropout = dropout
        self.gc1 = nn.Linear(in_ch, n_hid)
        self.gc2 = nn.Linear(n_hid, n_hid)
        self.gc3 = nn.Linear(n_hid, 1)

    def forward(self, x, G):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x))

        # Apply hypergraph convolution
        x = torch.matmul(G, x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x))

        # Apply second hypergraph convolution
        x = torch.matmul(G, x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x)

        # Ensure scores are between 0 and 1
        scores = torch.sigmoid(x)

        return scores.view(-1)


class HGNN_Ranking_Base(nn.Module):
    """
    Feature-less hypergraph neural network for ranking
    Only uses the graph structure (no node features)
    """

    def __init__(self, n_nodes, n_hid, dropout=0.5):
        super(HGNN_Ranking_Base, self).__init__()
        self.dropout = dropout
        self.n_nodes = n_nodes

        # Learnable node embeddings (instead of input features)
        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, n_hid))

        # Layers for processing embeddings
        self.gc1 = nn.Linear(n_hid, n_hid)
        self.gc2 = nn.Linear(n_hid, 1)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        # Initialize embeddings with normalized values
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self, G):
        # Start with learned node embeddings
        x = self.node_embeddings

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x))

        # Apply hypergraph convolution
        x = torch.matmul(G, x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)

        # Ensure scores are between 0 and 1
        scores = torch.sigmoid(x)

        return scores.view(-1)
