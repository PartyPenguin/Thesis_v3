import torch
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


class GCN_Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Policy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.proj_dim = hidden_dim
        self.num_heads = 6

        # Linear Layer for each node type
        self.joint_lin = Linear(input_dim, self.proj_dim)
        self.obj_lin = Linear(input_dim, self.proj_dim)
        self.tcp_lin = Linear(input_dim, self.proj_dim)
        self.goal_lin = Linear(input_dim, self.proj_dim)

        # List of GCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_dim * self.num_heads if i > 0 else self.proj_dim
            self.convs.append(
                GATv2Conv(
                    input_dim,
                    hidden_dim,
                    edge_dim=4,
                    heads=self.num_heads,
                    dropout=dropout,
                )
            )

        # Linear layer that projects the hidden dimension to the output dimension
        self.lin_out = Linear(hidden_dim * self.num_heads, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_type_mask = data.node_type_mask

        # Process each node type separately
        joint_x = self.joint_lin(x[node_type_mask == 0])
        tcp_x = self.tcp_lin(x[node_type_mask == 2])
        obj_x = self.obj_lin(x[node_type_mask == 1])
        goal_x = self.goal_lin(x[node_type_mask == 3])

        # Initialize an empty tensor to hold the processed features
        x_processed = torch.zeros(x.size(0), self.proj_dim, device=x.device)

        # Assign processed features to corresponding positions in x_processed
        x_processed[node_type_mask == 0] = joint_x
        x_processed[node_type_mask == 1] = obj_x
        x_processed[node_type_mask == 2] = tcp_x
        x_processed[node_type_mask == 3] = goal_x

        # Apply a ReLU activation
        x = F.relu(x_processed)

        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the GCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)

        # Project the hidden dimension to the output dimension
        x = self.lin_out(x)

        # Tanh activation for the output
        x = torch.tanh(x)

        return x


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(BaselineMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lins = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_dim if i == 0 else hidden_dim
            self.lins.append(Linear(input_dim, hidden_dim))
        self.lin_out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data

        for lin in self.lins:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_out(x)
        x = torch.tanh(x)

        return x