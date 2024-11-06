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


def create_pick_cube_graph(obs: np.ndarray) -> dict:
    # Convert observations to PyTorch tensor
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    # Truncate all values in obs to have maximal 5 decimal places
    batch_size = obs.shape[0]
    num_joints = 7
    # Define the total number of node types
    num_node_types = 4  # joint, tcp, obj, goal

    # ===============================
    # Split observation into components
    # ===============================
    joint_positions = obs[:, :9] 
    joint_velocities = obs[:, 9:18]
    is_grasped = obs[:, 18:19] 
    tcp_pose = obs[:, 19:26] 
    goal_position = obs[:, 26:29] 
    obj_pose = obs[:, 29:36] 
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]


    # ===============================
    # Compute forward kinematics for the batch
    # ===============================
    joint_se3_pos, joint_se3_rot = compute_fk(joint_positions[:,:7], end_only=False)

    # Take the first 7 joints for position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8]  # Shape: (batch_size, 7, 3)
    joint_se3_rot = joint_se3_rot[:, 1:8]  # Shape: (batch_size, 7, 4)
    joint_se3_pose = torch.cat([joint_se3_pos, joint_se3_rot], dim=-1)  # Shape: (batch_size, 7, 7)

    # ===============================
    # Create graph
    # ===============================

    # Step 1: Define one-hot encodings for each node type and expand them
    joint_one_hot = torch.tensor([1, 0, 0, 0], device=joint_positions.device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 4]
    tcp_one_hot = torch.tensor([0, 1, 0, 0], device=joint_positions.device).unsqueeze(0)  # Shape: [1, 4]
    obj_one_hot = torch.tensor([0, 0, 1, 0], device=joint_positions.device).unsqueeze(0)  # Shape: [1, 4]
    goal_one_hot = torch.tensor([0, 0, 0, 1], device=joint_positions.device).unsqueeze(0)  # Shape: [1, 4]

    # Expand one-hot encodings
    joint_one_hots = joint_one_hot.expand(batch_size, num_joints, -1)  # Shape: [batch_size, num_joints, 4]
    tcp_one_hot = tcp_one_hot.expand(batch_size, -1)  # Shape: [batch_size, 4]
    obj_one_hot = obj_one_hot.expand(batch_size, -1)  # Shape: [batch_size, 4]
    goal_one_hot = goal_one_hot.expand(batch_size, -1)  # Shape: [batch_size, 4]

    # Step 2: Prepare joint node features
    joint_positions_feat = joint_positions[:,:7].unsqueeze(-1)  # Shape: [batch_size, num_joints, 1]
    joint_velocities_feat = joint_velocities[:,:7].unsqueeze(-1)  # Shape: [batch_size, num_joints, 1]
    joint_se3_pose_feat = joint_se3_pose  # Shape: [batch_size, num_joints, feature_dim]

    joint_node_features = torch.cat([
        joint_positions_feat,
        joint_velocities_feat,
        joint_se3_pose_feat,
        joint_one_hots
    ], dim=2)  # Shape: [batch_size, num_joints, feature_dim + 2 + 4]

    # Step 3: Prepare TCP node features
    tcp_pose_feat = tcp_pose.unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
    tcp_to_obj_pos_feat = tcp_to_obj_pos.unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
    tcp_one_hot_feat = tcp_one_hot.unsqueeze(1)  # Shape: [batch_size, 1, 4]

    tcp_node_features = torch.cat([
        tcp_pose_feat,
        tcp_to_obj_pos_feat,
        tcp_one_hot_feat
    ], dim=2)  # Shape: [batch_size, 1, feature_dim * 2 + 4]

    # Step 4: Prepare object node features
    obj_pose_feat = obj_pose.unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
    obj_to_goal_pos_feat = obj_to_goal_pos.unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
    obj_one_hot_feat = obj_one_hot.unsqueeze(1)  # Shape: [batch_size, 1, 4]

    obj_node_features = torch.cat([
        obj_pose_feat,
        obj_to_goal_pos_feat,
        obj_one_hot_feat
    ], dim=2)  # Shape: [batch_size, 1, feature_dim * 2 + 4]

    # Step 5: Prepare goal node features
    goal_position_feat = goal_position.unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
    goal_one_hot_feat = goal_one_hot.unsqueeze(1)  # Shape: [batch_size, 1, 4]

    goal_node_features = torch.cat([
        goal_position_feat,
        goal_one_hot_feat
    ], dim=2)  # Shape: [batch_size, 1, feature_dim + 4]

    # Step 6: Pad features to the same dimensionality
    feature_dims = [
        joint_node_features.size(-1),
        tcp_node_features.size(-1),
        obj_node_features.size(-1),
        goal_node_features.size(-1)
    ]
    max_feature_dim = max(feature_dims)

    def pad_features(node, target_dim):
        pad_size = target_dim - node.size(-1)
        if pad_size > 0:
            padding = torch.zeros(node.size(0), node.size(1), pad_size, device=node.device)
            node = torch.cat([node, padding], dim=-1)
        return node

    joint_node_features = pad_features(joint_node_features, max_feature_dim)
    tcp_node_features = pad_features(tcp_node_features, max_feature_dim)
    obj_node_features = pad_features(obj_node_features, max_feature_dim)
    goal_node_features = pad_features(goal_node_features, max_feature_dim)

    # Step 7: Concatenate all node features
    node_features = torch.cat([
        joint_node_features,
        tcp_node_features,
        obj_node_features,
        goal_node_features
    ], dim=1)  # Shape: [batch_size, total_num_nodes, max_feature_dim]

    # Step 8: Create edges to form a [batch_size, 2, num_edges] tensor

    joint_node_indices = torch.arange(num_joints, device=device) # Shape: [num_joints]
    tcp_node_indices = torch.tensor([num_joints], device=device) # Shape: [1]
    obj_node_indices = torch.tensor([num_joints + 1], device=device) # Shape: [1]
    goal_node_indices = torch.tensor([num_joints + 2], device=device) # Shape: [1]

    # Connect all joints to each other joint
    joint_joint_edges = torch.combinations(joint_node_indices, 2).t()  # Shape: [2, num_edges]

    # Connect the TCP to all joints
    tcp_joint_edges = torch.stack([
        tcp_node_indices.expand(num_joints),
        joint_node_indices
    ], dim=0)  # Shape: [2, num_edges]

    # Connect the object to the TCP
    obj_tcp_edges = torch.stack([
        obj_node_indices.expand(num_joints),
        tcp_node_indices.expand(num_joints)
    ], dim=0)  # Shape: [2, num_edges]

    # Connect the goal to the object
    goal_obj_edges = torch.stack([
        goal_node_indices.expand(num_joints),
        obj_node_indices.expand(num_joints)
    ], dim=0)  # Shape: [2, num_edges]

    # Concatenate all edges
    edge_index = torch.cat([
        joint_joint_edges,
        tcp_joint_edges,
        obj_tcp_edges,
        goal_obj_edges
    ], dim=1)  # Shape: [2, total_num_edges]

    # Expand edge index to the batch dimension
    edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, 2, total_num_edges]



    # Step 9: Create the graph
    graph_list = []
    for i in range(batch_size):
        graph = Data(
            x=node_features[i],
            edge_index=edge_index[i],
            edge_attr=None
        )
        graph_list.append(graph)
    
    graph = Batch.from_data_list(graph_list)
    graph = T.ToUndirected()(graph) 

    visualize_graph(graph)

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
