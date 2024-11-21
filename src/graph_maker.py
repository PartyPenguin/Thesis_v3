from torch_geometric.data import Data, HeteroData
from torch_geometric.data import Batch
import torch 
import numpy as np
from tqdm import tqdm
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from util import compute_fk
from util import fourier_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_pick_cube_graph(obs: torch.tensor) -> dict:
    # Convert observations to PyTorch tensor and move to device
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs)
    obs = obs.to(device)

    batch_size = obs.shape[0]
    num_joints = 7
    num_node_types = 4  # joint, tcp, obj, goal

    # ===============================
    # Split observation into components
    # ===============================
    slices = {
        'joint_positions': (0, 9),
        'joint_velocities': (9, 18),
        'is_grasped': (18, 19),
        'tcp_pose': (19, 26),
        'goal_position': (26, 29),
        'obj_pose': (29, 36),
        'tcp_to_obj_pos': (36, 39),
        'obj_to_goal_pos': (39, 42),
    }
    obs_splits = {key: obs[:, start:end] for key, (start, end) in slices.items()}

    # ===============================
    # Compute forward kinematics for the batch
    # ===============================
    joint_se3_pos, joint_se3_rot = compute_fk(obs_splits['joint_positions'][:, :7], end_only=False)
    joint_se3_pose = torch.cat([joint_se3_pos[:, 1:8], joint_se3_rot[:, 1:8]], dim=-1)  # Shape: (batch_size, 7, 7)

    # ===============================
    # Create node features
    # ===============================
    node_types = {
        'joint': torch.tensor([1, 0, 0, 0], device=device),
        'tcp': torch.tensor([0, 1, 0, 0], device=device),
        'obj': torch.tensor([0, 0, 1, 0], device=device),
        'goal': torch.tensor([0, 0, 0, 1], device=device),
    }
    node_type_mask = []

    def create_node_features(features_list, one_hot_type, node_count):
        one_hot = node_types[one_hot_type].unsqueeze(0).unsqueeze(0).expand(batch_size, node_count, -1)
        features = torch.cat(features_list + [one_hot], dim=2)
        node_type_mask.append(torch.full((node_count,), len(node_type_mask), dtype=torch.long, device=device))
        return features

    joint_features = create_node_features(
        [joint_se3_pose, obs_splits['joint_positions'][:, :7].unsqueeze(-1)],
        'joint', num_joints
    )
    tcp_features = create_node_features(
        [
            obs_splits['tcp_pose'].unsqueeze(1),
            obs_splits['is_grasped'].unsqueeze(1),
            obs_splits['tcp_to_obj_pos'].unsqueeze(1)
        ],
        'tcp', 1
    )
    obj_features = create_node_features(
        [obs_splits['obj_pose'].unsqueeze(1), obs_splits['obj_to_goal_pos'].unsqueeze(1)],
        'obj', 1
    )
    goal_features = create_node_features(
        [obs_splits['goal_position'].unsqueeze(1)],
        'goal', 1
    )

    # Pad features to the same dimensionality
    max_feature_dim = max(f.size(-1) for f in [joint_features, tcp_features, obj_features, goal_features])
    pad = lambda x: torch.cat([x, torch.zeros(x.size(0), x.size(1), max_feature_dim - x.size(2), device=device)], dim=2)
    joint_features, tcp_features, obj_features, goal_features = map(pad, [joint_features, tcp_features, obj_features, goal_features])

    # Concatenate all node features
    node_features = torch.cat([joint_features, tcp_features, obj_features, goal_features], dim=1)
    node_type_mask = torch.cat(node_type_mask)

    # ===============================
    # Create edges
    # ===============================
    total_nodes = num_joints + 3  # joints + tcp + obj + goal
    indices = {
        'joint': torch.arange(num_joints, device=device),
        'tcp': torch.tensor([num_joints], device=device),
        'obj': torch.tensor([num_joints + 1], device=device),
        'goal': torch.tensor([num_joints + 2], device=device),
    }

    edge_list = []
    # Joint-to-joint edges
    joint_pairs = torch.combinations(indices['joint'], 2).t()
    edge_list.append(joint_pairs)
    # TCP-to-joint edges
    tcp_joint_edges = torch.stack([indices['tcp'].expand(num_joints), indices['joint']], dim=0)
    edge_list.append(tcp_joint_edges)
    # Object-to-TCP edge
    edge_list.append(torch.stack([indices['obj'], indices['tcp']]))
    # Goal-to-object edge
    edge_list.append(torch.stack([indices['goal'], indices['obj']]))

    edge_index = torch.cat(edge_list, dim=1).unsqueeze(0).expand(batch_size, -1, -1)

    # ===============================
    # Create the graph
    # ===============================
    graph_list = []
    for i in range(batch_size):
        edge_attr = compute_edge_attr(
            node_features[i][:, :3][edge_index[i][0]],
            node_features[i][:, :3][edge_index[i][1]]
        )
        graph = Data(
            x=node_features[i],
            edge_index=edge_index[i],
            edge_attr=edge_attr,
            node_type_mask=node_type_mask
        )
        graph_list.append(graph)

    graph = Batch.from_data_list(graph_list)
    graph = T.ToUndirected()(graph)

    return graph


            


def visualize_graph(graph:Data):
    # Visualize the graph using NetworkX and pyvis
    G = to_networkx(graph)
    
    # Create a Network object
    nt = Network()
    nt.from_nx(G)
    nt.show("graph.html", notebook=False)

def compute_edge_attr(src_pos, dst_pos):
    """
    Computes edge attributes given source and destination positions.

    Args:
        src_pos (Tensor): Source node positions, shape (E, 3)
        dst_pos (Tensor): Destination node positions, shape (E, 3)

    Returns:
        Tensor: Edge attributes, shape (E, 7)
    """

    vector = dst_pos - src_pos  # Shape: (E, 3)
    norm = torch.norm(vector, dim=1, keepdim=True)  # Shape: (E, 1)
    edge_attr = torch.cat([vector, norm], dim=1)  # Shape: (E, 4)
    return edge_attr
