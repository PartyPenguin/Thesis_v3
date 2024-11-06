import torch as th
from torch_geometric.nn import GATConv, GCNConv, HGTConv, HEATConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, Linear, BatchNorm

from torch_geometric.nn import HeteroConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import yaml


# Load config from params.yaml
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)


class GATPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims, config, node_types=None):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        num_heads = config["train"]["model_params"]["num_heads"]
        hidden_dim = config["train"]["model_params"]["hidden_dim"]
        dropout = config["train"]["model_params"]["dropout"]

        node_feature_emb = []
        node_feature_emb_dim = 8
        # for i in range(max(node_types) + 1):
        #     in_channel = node_types[node_types == i].shape[0]
        #     lin = Linear(in_channel, node_feature_emb_dim)
        #     node_feature_emb.append(lin)
        # self.node_feature_emb = nn.ModuleList(node_feature_emb)

        # Define the GAT layers
        self.gat_conv1 = GATConv(
            -1,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=5,
        )
        self.gat_conv2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=5,
        )
        self.gat_conv3 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=5,
        )

        self.lin = Linear(num_heads * hidden_dim, act_dims)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # node_types = node_types.view(batch.max() + 1, -1)
        # x_temp = x.clone()
        # x = []
        # for i in range(max(node_types[0]) + 1):
        #     x.append(self.node_feature_emb[i](x_temp[:, node_types[0] == i]))

        # x = th.cat(x, dim=1)

        x = self.gat_conv1(x, edge_index, edge_attr).relu()
        x = self.dropout(x)
        x = self.gat_conv2(x, edge_index, edge_attr).relu()
        x = self.dropout(x)
        x = self.gat_conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        x = th.tanh(x)

        return x



class HeteroGNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(HeteroGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers  # Number of GATConv layers
        self.heads = num_heads  # Number of attention heads
        self.dropout = dropout

        # Define linear layers for each node type to project features to hidden_dim
        self.node_proj = nn.ModuleDict({
            'joint': nn.Linear(7, hidden_dim),
            'tcp': nn.Linear(12, hidden_dim),
            'object': nn.Linear(10, hidden_dim),
            'goal': nn.Linear(3, hidden_dim)
        })
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_proj.keys()
        })

        # Define convolution layers for each relation type
        relations = [
            ('joint',  'kinematic',       'joint'),
            ('joint',  'kinematic',       'tcp'),
            ('joint',  'kinematic',       'object'),
            ('tcp',    'kinematic',       'object'),
            ('tcp',    'rev_kinematic',   'joint'),
            ('object', 'rev_kinematic',   'joint'),
            ('object', 'rev_kinematic',   'tcp'),
            ('joint',  'interaction',     'tcp'),
            ('tcp',    'interaction',     'object'),
            ('tcp',    'interaction',     'goal'),
            ('object', 'interaction',     'goal'),
            ('tcp',    'rev_interaction', 'joint'),
            ('object', 'rev_interaction', 'tcp'),
            ('goal',   'rev_interaction', 'tcp'),
            ('goal',   'rev_interaction', 'object'),
            ('object', 'proximity',       'object'),
            ('goal',   'proximity',       'goal'),
            ('joint',  'proximity',       'object'),
            ('joint',  'proximity',       'goal'),
            ('object', 'rev_proximity',   'joint'),
            ('goal',   'rev_proximity',   'joint'),
            # Add other edge types as needed
        ]

        # Create a list of HeteroConv layers
        self.convs = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            conv_dict = {}
            in_dim = hidden_dim if layer_idx == 0 else hidden_dim * self.heads
            for relation in relations:
                conv_dict[relation] = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=self.heads,
                    edge_dim=4,  # Edge attribute dimension
                    concat=True,
                    dropout=self.dropout,
                    add_self_loops=False
                )
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)

        # Define a final linear layer to map the graph representation to output_dim outputs
        self.output_layer = nn.Linear(hidden_dim * self.heads * len(self.node_proj), output_dim)

    def forward(self, data):
        # Project node features to hidden_dim
        x_dict = {}
        for node_type, x in data.x_dict.items():
            x = self.node_proj[node_type](x)
            x = F.relu(x)
            # x = self.norms[node_type](x)
            x_dict[node_type] = x

        # Apply convolution layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            # x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # ===============================
        # Perform pooling to get graph representations
        # ===============================
        pooled_embeddings = []

        for node_type, x in x_dict.items():
            batch = data[node_type].batch
            # Apply global mean pooling
            pooled = global_mean_pool(x, batch)  # Shape: (batch_size, hidden_dim)
            pooled_embeddings.append(pooled)

        # Concatenate per-node-type embeddings to get per-graph representations
        graph_representation = th.cat(pooled_embeddings, dim=1)  # Shape: (batch_size, hidden_dim * num_node_types)

        # Apply the final linear layer to map to output_dim outputs
        out = self.output_layer(graph_representation)  # Shape: (batch_size, output_dim)

        # Apply tanh activation
        out = th.tanh(out)

        # Return the outputs
        return out

class GCNPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims, config):
        super().__init__()
        self.obs_dims = obs_dims + 4
        self.act_dims = act_dims
        hidden_dim = config["train"]["model_params"]["hidden_dim"]
        dropout = config["train"]["model_params"]["dropout"]

        self.node_type_embeddings = nn.Embedding(4, 4)

        # Define the GCN layers
        self.gcn_conv1 = GCNConv(obs_dims + 4, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn_conv3 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        # Define the linear layer
        self.lin = Linear(hidden_dim, act_dims)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        node_types = th.tensor([0, 0, 0, 0, 0, 0, 0, 1, 2, 3]).to(x.device)
        node_type_emb = self.node_type_embeddings(node_types)
        node_type_emb = (
            node_type_emb.repeat(batch.max().item() + 1, 1)
            if batch is not None
            else node_type_emb
        )
        x = th.cat([x, node_type_emb], dim=-1)

        # Apply the GCN layers
        x = self.gcn_conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.gcn_conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.gcn_conv3(x, edge_index)

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        # Apply the linear layer
        x = self.lin(x)
        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        return x


class HGTPolicy(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()
        self.obs_dims = -1
        self.act_dims = out_channels

        self.conv1 = HGTConv(
            in_channels=-1,  # Automatically infer input dimensions
            out_channels=hidden_channels,
            metadata=data.metadata(),
            heads=num_heads,
        )
        self.conv2 = HGTConv(
            in_channels=-1,  # Automatically infer input dimensions
            out_channels=hidden_channels,
            metadata=data.metadata(),
            heads=num_heads,
        )
        self.conv3 = HGTConv(
            in_channels=-1,  # Automatically infer input dimensions
            out_channels=hidden_channels,
            metadata=data.metadata(),
            heads=num_heads,
        )

        # Since we're pooling and combining all node types,
        # the input dimension is hidden_channels * num_heads * num_node_types
        total_hidden_dim = hidden_channels
        self.lin = Linear(total_hidden_dim, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # Apply the convolutional layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Combine node embeddings from all node types
        x_all = th.cat([x for x in x_dict.values()], dim=0)
        batch_all = th.cat([batch for batch in batch_dict.values()], dim=0)

        # Pool over all nodes to get graph-level embeddings
        pooled = global_mean_pool(
            x_all, batch_all
        )  # Shape: [num_graphs, hidden_channels]

        # Pass through the linear layer and apply activation
        out = th.tanh(self.lin(pooled))  # Shape: [num_graphs, out_channels]

        return out


class HEATPolicy(nn.Module):

    def __init__(self, hidden_channels, out_channels, data):
        super().__init__()
        self.obs_dims = -1
        self.act_dims = out_channels
        num_edge_types = data.num_edge_types
        num_node_types = data.num_node_types

        dropout = config["train"]["model_params"]["dropout"]
        num_heads = config["train"]["model_params"]["num_heads"]

        self.conv1 = HEATConv(
            in_channels=-1,
            out_channels=hidden_channels,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            edge_type_emb_dim=8,
            edge_dim=4,
            edge_attr_emb_dim=64,
            heads=num_heads,
            dropout=dropout,
        )
        self.conv2 = HEATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            edge_type_emb_dim=8,
            edge_dim=4,
            edge_attr_emb_dim=64,
            heads=num_heads,
            dropout=dropout,
        )
        self.conv3 = HEATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            edge_type_emb_dim=8,
            edge_dim=4,
            edge_attr_emb_dim=64,
            heads=1,
            dropout=dropout,
        )
        self.lin = Linear(hidden_channels, 1)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        node_type = graph.node_type
        edge_type = graph.edge_type

        # Apply the convolutional layers
        x = self.conv1(x, edge_index, node_type, edge_type, edge_attr).relu()
        x = self.conv2(x, edge_index, node_type, edge_type, edge_attr).relu()
        x = self.conv3(x, edge_index, node_type, edge_type, edge_attr)

        # x = global_add_pool(x, graph.batch)
        # out = th.tanh(self.lin(x))
        # Return the action dim nodes for each batch
        x = th.tanh(self.lin(x))
        out = x.view(graph.batch.max() + 1, -1)[:, : self.act_dims]
        return out
