import torch
from torch_geometric.nn import GATConv, GCNConv, HGTConv, HEATConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, Linear, BatchNorm

from torch_geometric.nn import HeteroConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import MLP
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
        self.joint_lin = MLP(
            in_channels=input_dim,
            out_channels=self.proj_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
            act=nn.GELU(approximate='tanh'),
            norm="layer_norm",
        )
        self.obj_lin = MLP(
            in_channels=input_dim,
            out_channels=self.proj_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
            act=nn.GELU(approximate='tanh'),
            norm="layer_norm",
        )
        self.tcp_lin = MLP(
            in_channels=input_dim,
            out_channels=self.proj_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
            act=nn.GELU(approximate='tanh'),
            norm="layer_norm",
        )
        self.goal_lin = MLP(        
            in_channels=input_dim,
            out_channels=self.proj_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
            act=nn.GELU(approximate='tanh'),
            norm="layer_norm",
        )

        # List of GCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_dim * self.num_heads if i > 0 else self.proj_dim
            self.convs.append(
                GATv2Conv(
                    input_dim,
                    hidden_dim,
                    edge_dim=54,
                    heads=self.num_heads,
                    dropout=dropout,
                )
            )

        # Seperate prediction head for translation and grasp
        self.prediction_head = MLP(
            in_channels=hidden_dim * self.num_heads,
            out_channels=output_dim - 1,
            hidden_channels=hidden_dim,
            num_layers=2,
            act=nn.GELU(approximate='tanh'),
            norm="layer_norm",
        )
        self.prediction_head_grasp = MLP(
            in_channels=hidden_dim * self.num_heads,
            out_channels=1,
            hidden_channels=hidden_dim,
            num_layers=2,
            act=nn.GELU(approximate='tanh'),
            norm="layer_norm",
        )

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
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the GCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling
        # x = global_mean_pool(x, data.batch)

        # Only take joint and tcp nodes for prediction
        x = x[torch.logical_or(node_type_mask == 0, node_type_mask == 2)]

        x = global_mean_pool(x, data.batch[torch.logical_or(node_type_mask == 0, node_type_mask == 2)])

        # Project the hidden dimension to the output dimension
        x_q = self.prediction_head(x)
        x_g = self.prediction_head_grasp(x)
        
        # Concatenate the translation and grasp predictions
        x = torch.cat([x_q, x_g], dim=1)

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
