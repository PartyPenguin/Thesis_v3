from torch_geometric.data import Data, HeteroData
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_pick_cube_graph_batched_old(obs: np.ndarray):
    batch_size = obs.shape[0]
    # The offset of the robot in the scene must be added to the joint positions
    offset = th.tensor([-0.615, 0, 0]).to(device)

    # Split observation into components
    joint_positions = obs[:, :9]  # Shape: (batch_size, 9)
    joint_velocities = obs[:, 9:18]  # Shape: (batch_size, 9)
    is_grasped = obs[:, 18:19]  # Shape: (batch_size, 1)
    tcp_pose = obs[:, 19:26]  # Shape: (batch_size, 7)
    goal_position = obs[:, 26:29]  # Shape: (batch_size, 3)
    obj_pose = obs[:, 29:36]  # Shape: (batch_size, 7)
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    # Compute forward kinematics for the batch
    joint_se3_pos, _ = compute_fk(joint_positions, end_only=False)
    joint_se3_pos = joint_se3_pos
    if joint_se3_pos.dim() == 2:
        joint_se3_pos = joint_se3_pos.unsqueeze(0)
    else:
        joint_se3_pos = joint_se3_pos.permute(1, 0, 2)

    # Take the first 7 joints for SE(3) position and rotation
    joint_se3_pos = joint_se3_pos[:, :8].detach().cpu().numpy()

    nodes = np.concatenate(
        (
            joint_se3_pos,
            joint_positions[:, :8, np.newaxis],
            joint_velocities[:, :8, np.newaxis],
            np.repeat(tcp_pose[:, np.newaxis, :], 8, axis=1),
            np.repeat(is_grasped[:, np.newaxis, :], 8, axis=1),
            np.repeat(goal_position[:, np.newaxis, :], 8, axis=1),
            np.repeat(obj_pose[:, np.newaxis, :], 8, axis=1),
            np.repeat(tcp_to_obj_pos[:, np.newaxis, :], 8, axis=1),
            np.repeat(obj_to_goal_pos[:, np.newaxis, :], 8, axis=1),
        ),
        axis=-1,
    )

    edge_list = [[i, i + 1] for i in range(7)]
    # Skip connection from joitn 2 to 5
    edge_list = edge_list + [[1, 4]]
    edge_list = np.array(edge_list)

    edge_attr = np.linalg.norm(np.diff(nodes[:, :, :3], axis=1), axis=-1)
    edge_attr_skip_connection = np.linalg.norm(
        nodes[:, 1, :3] - nodes[:, 4, :3], axis=-1
    )
    edge_attr = np.concatenate(
        [edge_attr, edge_attr_skip_connection[:, np.newaxis]], axis=-1
    )

    graph_list = [
        T.ToUndirected()(
            Data(
                x=th.tensor(nodes[i]).float(),
                edge_index=th.tensor(edge_list).long().T,
                edge_attr=th.tensor(edge_attr[i]).float(),
            )
        )
        for i in range(batch_size)
    ]
    return graph_list


def create_pick_cube_graph_batched(obs: np.ndarray):
    from util import compute_fk, rot_normalize
    batch_size = obs.shape[0]
    # The offset of the robot in the scene must be added to the joint positions
    offset = th.tensor([-0.615, 0, 0]).to(device)

    # Split observation into components
    joint_positions = obs[:, :9]  # Shape: (batch_size, 9)
    joint_velocities = obs[:, 9:18]  # Shape: (batch_size, 9)
    is_grasped = obs[:, 18:19]  # Shape: (batch_size, 1)
    tcp_pose = obs[:, 19:26]  # Shape: (batch_size, 7)
    goal_position = obs[:, 26:29]  # Shape: (batch_size, 3)
    obj_pose = obs[:, 29:36]  # Shape: (batch_size, 7)
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    # Compute forward kinematics for the batch
    joint_se3_pos, joint_se3_rot = compute_fk(joint_positions, end_only=False)
    joint_se3_pos = joint_se3_pos + offset
    if joint_se3_pos.dim() == 2:
        joint_se3_pos = joint_se3_pos.unsqueeze(0)
        joint_se3_rot = joint_se3_rot.unsqueeze(0)
    else:
        joint_se3_pos = joint_se3_pos.permute(1, 0, 2)
        joint_se3_rot = joint_se3_rot.permute(1, 0, 2)

    # Take the first 7 joints for SE(3) position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8].detach().cpu().numpy()
    # Shape: (batch_size, 7, 3)
    joint_se3_rot = rot_normalize(joint_se3_rot[:, 1:8].detach().cpu().numpy())
    # Shape: (batch_size, 7, 4)

    # Node Types
    # -------------------------------
    node_type_list = [0] * 7  # Joint nodes
    node_type_list.extend([1] * 1)  # TCP node
    node_type_list.extend([2] * 1)  # Object node
    node_type_list.extend([3] * 1)  # Goal node
    node_type_list = np.array(node_type_list)
    node_type_embeddings = (
        nn.functional.one_hot(th.tensor(node_type_list), num_classes=4)
        .float()
        .detach()
        .numpy()
    )
    node_type_embeddings = np.repeat(
        node_type_embeddings[np.newaxis, :], batch_size, axis=0
    )

    # Building Nodes
    # -------------------------------
    # Combine positions and rotations
    joint_se3_pose = np.concatenate(
        [joint_se3_pos], axis=-1
    )  # Shape: (batch_size, 7, 7)
    # Combine joint positions, velocities, and SE(3) pose to create joint nodes
    joint_nodes = np.concatenate(
        [
            joint_se3_pose,  # (batch_size, 7, 7)
            # joint_positions[:, :7, np.newaxis],  # (batch_size, 7, 1)
            # joint_velocities[:, :7, np.newaxis],  # (batch_size, 7, 1)
            # np.repeat(tcp_pose[:, np.newaxis, :], 7, axis=1),  # (batch_size, 7, 7)
            # np.repeat(is_grasped[:, np.newaxis, :], 7, axis=1),  # (batch_size, 7, 1)
            # np.repeat(goal_position[:, np.newaxis, :], 7, axis=1),  # (batch_size, 7, 3)
            # np.repeat(obj_pose[:, np.newaxis, :], 7, axis=1),  # (batch_size, 7, 7)
            node_type_embeddings[:, :7],  # (batch_size, 7, 4)
        ],
        axis=-1,
    )  # Shape: (batch_size, 7, 9)

    # Create TCP node
    tcp_node = np.concatenate(
        [tcp_pose, is_grasped, node_type_embeddings[:, 7]], axis=-1
    )[:, np.newaxis, :]

    # Create object node
    object_node = np.concatenate([obj_pose, node_type_embeddings[:, 8]], axis=-1)[
        :, np.newaxis, :
    ]

    # Create goal node
    goal_node = np.concatenate([goal_position, node_type_embeddings[:, 9]], axis=-1)[
        :, np.newaxis, :
    ]

    # Pad nodes to the same size
    max_nodes = max(
        joint_nodes.shape[-1],
        tcp_node.shape[-1],
        object_node.shape[-1],
        goal_node.shape[-1],
    )
    joint_nodes = np.pad(
        joint_nodes,
        ((0, 0), (0, 0), (0, max_nodes - joint_nodes.shape[-1])),
        mode="constant",
        constant_values=0,
    )
    tcp_node = np.pad(
        tcp_node,
        ((0, 0), (0, 0), (0, max_nodes - tcp_node.shape[-1])),
        mode="constant",
        constant_values=0,
    )
    object_node = np.pad(
        object_node,
        ((0, 0), (0, 0), (0, max_nodes - object_node.shape[-1])),
        mode="constant",
        constant_values=0,
    )
    goal_node = np.pad(
        goal_node,
        ((0, 0), (0, 0), (0, max_nodes - goal_node.shape[-1])),
        mode="constant",
        constant_values=0,
    )

    # Stack all nodes
    nodes = np.concatenate([joint_nodes, tcp_node, object_node, goal_node], axis=1)

    # Building Edges
    # -------------------------------
    # Define edges
    edge_type_list = []
    # Joint edges
    # *************
    joint_joint_edges = [[i, i + 1] for i in range(6)]
    joint_joint_edges.append(joint_joint_edges[-1])
    edge_type_list.extend([0] * len(joint_joint_edges))

    # TCP edges
    # *************
    tcp_joint_edges = [[7, i] for i in range(7)]
    tcp_joint_edges.append(tcp_joint_edges[-1])
    edge_type_list.extend([1] * len(tcp_joint_edges))

    # Object edges
    # *************
    object_tcp_edges = [[8, 7]]
    edge_type_list.extend([2] * len(object_tcp_edges))

    # Goal edges
    # *************
    goal_object_edges = [[9, 8]]
    edge_type_list.extend([3] * len(goal_object_edges))

    edges = np.concatenate(
        [
            joint_joint_edges,
            tcp_joint_edges,
            object_tcp_edges,
            goal_object_edges,
        ],
        axis=0,
    )
    edges = np.array(edges)  # Shape: (number_of_edges, 2)

    # Now create the edge list and attribute list for each element in the batch
    graphs = []
    edge_type_encoding = nn.Embedding(4, 4).to(device)
    edge_encoding = edge_type_encoding(
        th.tensor(np.array([edge_type_list])).to(device)
    ).squeeze()

    for batch_idx in range(batch_size):
        # Select nodes and edges for this batch
        nodes_batch = nodes[batch_idx]

        # Calculate edge attributes (distances between nodes in SE(3) space)
        edge_attributes = []
        for i, edge in enumerate(edges):
            node1 = nodes_batch[edge[0], :3]  # Extract position (3-dimensional)
            node2 = nodes_batch[edge[1], :3]  # Extract position (3-dimensional)
            rel_pos = node2 - node1
            attr = np.concatenate(
                [[np.linalg.norm(rel_pos)], edge_encoding[i].detach().cpu().numpy()]
            )
            edge_attributes.append(attr)
        # Convert to tensors
        edge_attributes = th.tensor(edge_attributes).float()
        edge_index = th.tensor(edges).long().T
        node_features = th.tensor(nodes_batch).float()

        # Create the graph for this batch element
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)

        # Convert to NetworkX graph
        # G = to_networkx(graph)

        # # # Extract 3D positions
        # pos = {i: nodes_batch[i, 0:3] for i in range(nodes_batch.shape[0])}

        # # Plot the graph in 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # for i, edge in enumerate(edges):
        #     x = [pos[edge[0]][0], pos[edge[1]][0]]
        #     y = [pos[edge[0]][1], pos[edge[1]][1]]
        #     z = [pos[edge[0]][2], pos[edge[1]][2]]
        #     # Color based on edge type
        #     if edge_type_list[i] == 0:
        #         ax.plot(x, y, z, color="r")
        #     elif edge_type_list[i] == 1:
        #         ax.plot(x, y, z, color="g")
        #     elif edge_type_list[i] == 2:
        #         ax.plot(x, y, z, color="y")
        #     elif edge_type_list[i] == 3:
        #         ax.plot(x, y, z, color="b")
        #     elif edge_type_list[i] == 4:
        #         ax.plot(x, y, z, color="k")
        #     # ax.plot(x, y, z, color="r")

        # for node in G.nodes():
        #     ax.scatter(pos[node][0], pos[node][1], pos[node][2], color="r")
        #     # Add node id's
        #     ax.text(pos[node][0], pos[node][1], pos[node][2], s=str(node), fontsize=12)

        # plt.show()

        # Append to the list of graphs
        graphs.append(graph)

    return graphs


def create_hetero_pick_cube_graph_batched_old_old(obs: np.ndarray):
    batch_size = obs.shape[0]

    # The offset of the robot in the scene must be added to the joint positions
    offset = th.tensor([-0.615, 0, 0]).to(device)

    # Split observation into components
    joint_positions = obs[:, :9]  # Shape: (batch_size, 9)
    joint_velocities = obs[:, 9:18]  # Shape: (batch_size, 9)
    is_grasped = obs[:, 18:19]  # Shape: (batch_size, 1)
    tcp_pose = obs[:, 19:26]  # Shape: (batch_size, 7)
    goal_position = obs[:, 26:29]  # Shape: (batch_size, 3)
    obj_pose = obs[:, 29:36]  # Shape: (batch_size, 7)
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    # Compute forward kinematics for the batch
    joint_se3_pos, _ = compute_fk(joint_positions, end_only=False)
    joint_se3_pos = joint_se3_pos + offset
    if joint_se3_pos.dim() == 2:
        joint_se3_pos = joint_se3_pos.unsqueeze(0)
    else:
        joint_se3_pos = joint_se3_pos.permute(1, 0, 2)

    # Take the first 7 joints for SE(3) position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8].detach().cpu().numpy()

    # Joint Nodes
    # -------------------------------
    joint_features = np.concatenate(
        [
            joint_se3_pos,
            joint_positions[:, :7, np.newaxis],
            joint_velocities[:, :7, np.newaxis],
        ],
        axis=-1,
    )

    # TCP Node
    # -------------------------------
    tcp_features = np.concatenate([tcp_pose, is_grasped], axis=-1)

    # Object Node
    # -------------------------------
    object_features = obj_pose

    # Goal Node
    # -------------------------------
    goal_features = goal_position

    # Global Node
    # -------------------------------
    global_features = np.concatenate([goal_position, obj_pose, is_grasped], axis=-1)

    # Edges
    # -------------------------------
    joint_joint_edges = [[i, i + 1] for i in range(6)]
    joint_joint_edges = joint_joint_edges + [[i + 1, i] for i in range(6)]

    joint_tcp_edges = [[6, 0]]
    tcp_joint_edges = [[0, 6]]

    object_tcp_edges = [[0, 0]]
    tcp_object_edges = [[0, 0]]

    goal_object_edges = [[0, 0]]

    global_joint_edges = [[0, i] for i in range(7)]
    joint_global_edges = [[i, 0] for i in range(7)]
    global_tcp_edges = [[0, 0]]
    tcp_global_edges = [[0, 0]]
    global_object_edges = [[0, 0]]
    object_global_edges = [[0, 0]]
    global_goal_edges = [[0, 0]]
    goal_global_edges = [[0, 0]]

    # Edge Attributes
    # -------------------------------
    # Calculate edge attributes (distances between nodes in SE(3) space)
    joint_joint_edge_attributes = []
    for edge in joint_joint_edges:
        node1 = joint_features[:, edge[0], :3]  # Extract position (3-dimensional)
        node2 = joint_features[:, edge[1], :3]  # Extract position (3-dimensional)
        rel_pos = node2 - node1
        dist = np.linalg.norm(rel_pos, axis=-1)
        attr = np.concatenate([rel_pos, dist[:, np.newaxis]], axis=-1)
        joint_joint_edge_attributes.append(attr)
    joint_joint_edge_attributes = np.array(joint_joint_edge_attributes).T.transpose(
        1, 2, 0
    )

    tcp_joint_edge_attributes = []
    for edge in joint_tcp_edges:
        node1 = tcp_features[:, :3]  # Extract position (3-dimensional)
        node2 = joint_features[:, edge[1], :3]  # Extract position (3-dimensional)
        rel_pos = node2 - node1
        dist = np.linalg.norm(rel_pos, axis=-1)
        attr = np.concatenate([rel_pos, dist[:, np.newaxis]], axis=-1)
        tcp_joint_edge_attributes.append(attr)
    tcp_joint_edge_attributes = np.array(tcp_joint_edge_attributes).T.transpose(1, 2, 0)

    joint_tcp_edge_attributes = []
    for edge in joint_tcp_edges:
        node1 = joint_features[:, edge[0], :3]
        node2 = tcp_features[:, :3]
        rel_pos = node2 - node1
        dist = np.linalg.norm(rel_pos, axis=-1)
        attr = np.concatenate([rel_pos, dist[:, np.newaxis]], axis=-1)
        joint_tcp_edge_attributes.append(attr)
    joint_tcp_edge_attributes = np.array(joint_tcp_edge_attributes).T.transpose(1, 2, 0)

    object_tcp_edge_attributes = []
    for edge in object_tcp_edges:
        node1 = object_features[:, :3]
        node2 = tcp_features[:, :3]
        rel_pos = node2 - node1
        dist = np.linalg.norm(rel_pos, axis=-1)
        attr = np.concatenate([rel_pos, dist[:, np.newaxis]], axis=-1)
        object_tcp_edge_attributes.append(attr)
    object_tcp_edge_attributes = np.array(object_tcp_edge_attributes).T.transpose(
        1, 2, 0
    )

    tcp_object_edge_attributes = []
    for edge in tcp_object_edges:
        node1 = goal_features
        node2 = tcp_features[:, :3]
        rel_pos = node2 - node1
        dist = np.linalg.norm(rel_pos, axis=-1)
        attr = np.concatenate([rel_pos, dist[:, np.newaxis]], axis=-1)
        tcp_object_edge_attributes.append(attr)
    tcp_object_edge_attributes = np.array(tcp_object_edge_attributes).T.transpose(
        1, 2, 0
    )

    goal_object_edge_attributes = []
    for edge in goal_object_edges:
        node1 = object_features[:, :3]
        node2 = goal_features
        rel_pos = node2 - node1
        dist = np.linalg.norm(rel_pos, axis=-1)
        attr = np.concatenate([rel_pos, dist[:, np.newaxis]], axis=-1)
        goal_object_edge_attributes.append(attr)
    goal_object_edge_attributes = np.array(goal_object_edge_attributes).T.transpose(
        1, 2, 0
    )

    # Global Edges initialized with ones
    global_joint_edge_attributes = np.ones((batch_size, 7, 4))
    joint_global_edge_attributes = np.ones((batch_size, 7, 4))
    global_tcp_edge_attributes = np.ones((batch_size, 1, 4))
    tcp_global_edge_attributes = np.ones((batch_size, 1, 4))
    global_object_edge_attributes = np.ones((batch_size, 1, 4))
    object_global_edge_attributes = np.ones((batch_size, 1, 4))
    global_goal_edge_attributes = np.ones((batch_size, 1, 4))
    goal_global_edge_attributes = np.ones((batch_size, 1, 4))

    # Create the graph
    # -------------------------------
    graph_list = []
    for batch in range(batch_size):
        data = HeteroData()
        data["joint"].x = th.tensor(joint_features[batch]).float()
        data["tcp"].x = th.tensor(tcp_features[batch]).float().unsqueeze(0)
        data["object"].x = th.tensor(object_features[batch]).float().unsqueeze(0)
        data["goal"].x = th.tensor(goal_features[batch]).float().unsqueeze(0)
        data["global"].x = th.tensor(global_features[batch]).float().unsqueeze(0)

        # Edge Index

        data["joint", "connects", "joint"].edge_index = (
            th.tensor(joint_joint_edges).long().T
        )
        data["joint", "connects", "tcp"].edge_index = (
            th.tensor(joint_tcp_edges).long().T
        )
        data["tcp", "connects", "joint"].edge_index = (
            th.tensor(tcp_joint_edges).long().T
        )
        data["object", "connects", "tcp"].edge_index = (
            th.tensor(object_tcp_edges).long().T
        )
        data["tcp", "connects", "object"].edge_index = (
            th.tensor(tcp_object_edges).long().T
        )
        data["goal", "connects", "object"].edge_index = (
            th.tensor(goal_object_edges).long().T
        )
        data["global", "connects", "joint"].edge_index = (
            th.tensor(global_joint_edges).long().T
        )
        data["joint", "connects", "global"].edge_index = (
            th.tensor(joint_global_edges).long().T
        )
        data["global", "connects", "tcp"].edge_index = (
            th.tensor(global_tcp_edges).long().T
        )
        data["tcp", "connects", "global"].edge_index = (
            th.tensor(tcp_global_edges).long().T
        )
        data["global", "connects", "object"].edge_index = (
            th.tensor(global_object_edges).long().T
        )
        data["object", "connects", "global"].edge_index = (
            th.tensor(object_global_edges).long().T
        )
        data["global", "connects", "goal"].edge_index = (
            th.tensor(global_goal_edges).long().T
        )
        data["goal", "connects", "global"].edge_index = (
            th.tensor(goal_global_edges).long().T
        )

        # Edge Attributes

        data["joint", "connects", "joint"].edge_attr = th.tensor(
            joint_joint_edge_attributes[batch]
        ).float()

        data["joint", "connects", "tcp"].edge_attr = th.tensor(
            tcp_joint_edge_attributes[batch]
        ).float()
        data["tcp", "connects", "joint"].edge_attr = th.tensor(
            joint_tcp_edge_attributes[batch]
        ).float()
        data["object", "connects", "tcp"].edge_attr = th.tensor(
            object_tcp_edge_attributes[batch]
        ).float()
        data["tcp", "connects", "object"].edge_attr = th.tensor(
            tcp_object_edge_attributes[batch]
        ).float()
        data["goal", "connects", "object"].edge_attr = th.tensor(
            goal_object_edge_attributes[batch]
        ).float()
        data["global", "connects", "joint"].edge_attr = th.tensor(
            global_joint_edge_attributes[batch]
        ).float()
        data["joint", "connects", "global"].edge_attr = th.tensor(
            joint_global_edge_attributes[batch]
        ).float()
        data["global", "connects", "tcp"].edge_attr = th.tensor(
            global_tcp_edge_attributes[batch]
        ).float()
        data["tcp", "connects", "global"].edge_attr = th.tensor(
            tcp_global_edge_attributes[batch]
        ).float()
        data["global", "connects", "object"].edge_attr = th.tensor(
            global_object_edge_attributes[batch]
        ).float()
        data["object", "connects", "global"].edge_attr = th.tensor(
            object_global_edge_attributes[batch]
        ).float()
        data["global", "connects", "goal"].edge_attr = th.tensor(
            global_goal_edge_attributes[batch]
        ).float()
        data["goal", "connects", "global"].edge_attr = th.tensor(
            goal_global_edge_attributes[batch]
        ).float()

        data = T.AddSelfLoops(attr="edge_attr", fill_value=0)(data)
        data = data.to_homogeneous()
        data.cpu()
        graph_list.append(data)

        # x = to_networkx(data)
        # nx.draw(x, with_labels=True)
        # plt.show()

    return graph_list


def create_hetero_pick_cube_graph_batched_old(obs: np.ndarray):
    obs = th.tensor(obs).float()
    batch_size = obs.shape[0]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    offset = th.tensor([-0.615, 0, 0], device=device)

    # Split observation into components
    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:18]
    is_grasped = obs[:, 18:19]
    tcp_pose = obs[:, 19:26]
    goal_position = obs[:, 26:29]
    obj_pose = obs[:, 29:36]
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    # Compute forward kinematics for the batch
    joint_se3_pos, joint_se3_rot = compute_fk(joint_positions, end_only=False)
    joint_se3_pos += offset

    # Ensure correct tensor dimensions
    if joint_se3_pos.dim() == 2:
        joint_se3_pos = joint_se3_pos.unsqueeze(0)
        joint_se3_rot = joint_se3_rot.unsqueeze(0)
    else:
        joint_se3_pos = joint_se3_pos.permute(1, 0, 2)
        joint_se3_rot = joint_se3_rot.permute(1, 0, 2)

    # Take the first 7 joints for position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8].detach().cpu()
    joint_se3_rot = joint_se3_rot[:, 1:8].detach().cpu()
    joint_se3_pose = th.cat([joint_se3_pos, joint_se3_rot], axis=-1)

    # Node Types (one-hot encoding)
    node_type_joint = th.tensor([1, 0, 0, 0, 0])
    node_type_tcp = th.tensor([0, 1, 0, 0, 0])
    node_type_global = th.tensor([0, 0, 1, 0, 0])
    node_type_object = th.tensor([0, 0, 0, 1, 0])
    node_type_goal = th.tensor([0, 0, 0, 0, 1])

    # Define fixed sizes for each feature type
    NODE_TYPE_SIZE = 5
    NODE_POSE_SIZE = 7  # SE(3) pose (3D position + 3D rotation)
    JOINT_POS_SIZE = 1  # Joint positions
    JOINT_VEL_SIZE = 1  # Joint velocities
    GRASP_SIZE = 1
    OBJ_POSE_SIZE = 7  # SE(3) pose (3D position + 3D rotation)
    GOAL_POSITION_SIZE = 3  # 3D position

    node_types = th.cat(
        [
            th.tensor([0]).repeat(NODE_TYPE_SIZE),
            th.tensor([1]).repeat(NODE_POSE_SIZE),
            th.tensor([2]).repeat(JOINT_POS_SIZE),
            th.tensor([3]).repeat(JOINT_VEL_SIZE),
            th.tensor([4]).repeat(GRASP_SIZE),
            th.tensor([5]).repeat(OBJ_POSE_SIZE),
            th.tensor([6]).repeat(GOAL_POSITION_SIZE),
        ]
    )

    # ===============================
    # Node Features
    # ===============================

    # -------------------------------
    # Prepare joint nodes
    joint_node = th.cat(
        [
            node_type_joint.repeat(batch_size, 7, 1),
            th.tensor(joint_se3_pose),
            joint_positions[:, :7].unsqueeze(-1),
            joint_velocities[:, :7].unsqueeze(-1),
            th.rand(batch_size, 7, GRASP_SIZE),
            th.rand(batch_size, 7, OBJ_POSE_SIZE),
            th.rand(batch_size, 7, GOAL_POSITION_SIZE),
        ],
        dim=-1,
    )

    # -------------------------------
    # Prepare TCP node
    tcp_node = th.cat(
        [
            node_type_tcp.repeat(batch_size, 1, 1),
            tcp_pose.unsqueeze(1),
            joint_positions[:, 8:9].unsqueeze(-1),
            th.rand(batch_size, 1, JOINT_VEL_SIZE),
            is_grasped.unsqueeze(1),
            th.rand(batch_size, 1, OBJ_POSE_SIZE),
            th.rand(batch_size, 1, GOAL_POSITION_SIZE),
        ],
        axis=-1,
    )

    # -------------------------------
    # Prepare Object node
    object_node = th.cat(
        [
            node_type_object.repeat(batch_size, 1, 1),
            th.rand(batch_size, 1, NODE_POSE_SIZE),
            th.rand(batch_size, 1, JOINT_POS_SIZE),
            th.rand(batch_size, 1, JOINT_VEL_SIZE),
            th.rand(batch_size, 1, GRASP_SIZE),
            obj_pose.unsqueeze(1),
            th.rand(batch_size, 1, GOAL_POSITION_SIZE),
        ],
        axis=-1,
    )

    # -------------------------------
    # Prepare Goal node
    goal_node = th.cat(
        [
            node_type_goal.repeat(batch_size, 1, 1),
            th.rand(batch_size, 1, NODE_POSE_SIZE),
            th.rand(batch_size, 1, JOINT_POS_SIZE),
            th.rand(batch_size, 1, JOINT_VEL_SIZE),
            th.rand(batch_size, 1, GRASP_SIZE),
            th.rand(batch_size, 1, OBJ_POSE_SIZE),
            goal_position.unsqueeze(1),
        ],
        axis=-1,
    )

    # -------------------------------
    # Combine all nodes into a single tensor
    # All nodes have the same dimension now after padding
    nodes = th.cat([joint_node, tcp_node, object_node, goal_node], axis=1)

    # Node indices
    # Joint nodes: indices 0-6
    # TCP node: index 7
    # Object node: index 8
    # Goal node: index 9

    # ===============================
    # Edge Features
    # ===============================

    # -------------------------------
    # Joint-Joint Edges
    joint_joint_edges = th.tensor(
        [[i, i + 1] for i in range(6)] + [[i + 1, i] for i in range(6)], dtype=int
    )
    dist = th.diff(joint_se3_pose[:, :, :3], axis=1)
    rev_dist = -dist
    norm = th.linalg.norm(dist, axis=-1, keepdims=True)
    joint_joint_edge_attr = th.cat(
        [
            th.cat([dist, rev_dist], axis=1),
            th.cat([norm, norm], axis=1),
        ],
        axis=-1,
    )

    # -------------------------------
    # TCP-Joint Edges
    tcp_joint_edges = th.tensor(
        [[7, i] for i in range(7)] + [[i, 7] for i in range(7)], dtype=th.long
    )
    dist = joint_se3_pose[:, :7, :3] - tcp_pose[:, None, :3]
    rev_dist = -dist
    norm_tcp_joint = th.linalg.norm(dist, dim=-1, keepdim=True)
    tcp_joint_edge_attr = th.cat(
        [
            th.cat([dist, rev_dist], dim=1),
            th.cat([norm_tcp_joint, norm_tcp_joint], dim=1),
        ],
        dim=-1,
    )

    # -------------------------------
    # Object-TCP Edges
    object_tcp_edges = th.tensor([[8, 7], [7, 8]], dtype=th.long)
    dist = tcp_pose[:, :3].unsqueeze(1) - obj_pose[:, :3].unsqueeze(1)
    rev_dist = -dist
    norm = th.linalg.norm(dist, dim=-1, keepdim=True)
    object_tcp_edge_attr = th.cat(
        [
            th.cat([dist, rev_dist], dim=1),
            th.cat([norm, norm], dim=1),
        ],
        dim=-1,
    )

    # -------------------------------
    # Goal-TCP Edges
    goal_tcp_edges = th.tensor([[9, 7], [7, 9]], dtype=th.long)
    dist = tcp_pose[:, :3].unsqueeze(1) - goal_position.unsqueeze(1)
    rev_dist = -dist
    norm = th.linalg.norm(dist, dim=-1, keepdim=True)
    goal_tcp_edge_attr = th.cat(
        [
            th.cat([dist, rev_dist], dim=1),
            th.cat([norm, norm], dim=1),
        ],
        dim=-1,
    )

    # -------------------------------
    # Object-Goal Edges
    object_goal_edge = th.tensor([[8, 9], [9, 8]], dtype=th.long)
    dist = obj_to_goal_pos.unsqueeze(1)
    rev_dist = -dist
    norm = th.linalg.norm(dist, dim=-1, keepdim=True)
    object_goal_edge_attr = th.cat(
        [
            th.cat([dist, rev_dist], dim=1),
            th.cat([norm, norm], dim=1),
        ],
        dim=-1,
    )

    # Combine all edges
    edge_index = th.cat(
        [
            joint_joint_edges,
            tcp_joint_edges,
            object_tcp_edges,
            goal_tcp_edges,
            object_goal_edge,
        ],
        dim=0,
    )

    # Prepare edge attributes
    edge_attr = th.cat(
        [
            joint_joint_edge_attr,
            tcp_joint_edge_attr,
            object_tcp_edge_attr,
            goal_tcp_edge_attr,
            object_goal_edge_attr,
        ],
        dim=1,
    )

    # Create the graph
    graph_list = []
    for i in range(batch_size):
        data = Data(
            x=nodes[i],
            edge_index=edge_index.T,
            edge_attr=edge_attr[i],
            node_types=node_types,
        )
        data = T.AddSelfLoops(attr="edge_attr", fill_value=0)(data)
        data = data.to(device)
        graph_list.append(data)

        # x = to_networkx(data)
        # nx.draw(x, with_labels=True)
        # plt.show()

    return graph_list


def create_hetero_pick_cube_graph_batched(obs: np.ndarray, verbose=False, noise=0):
    """
    Creates a batched HeteroData object for imitation learning using GNNs.
    The graph represents the state of a robot manipulation task involving a
    Franka Panda robot, an object, and a goal location.

    Args:
        obs (np.ndarray): The observation array of shape (batch_size, feature_dim).

    Returns:
        HeteroData: A PyTorch Geometric HeteroData object representing the batched graphs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert observations to PyTorch tensor
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    # Truncate all values in obs to have maximal 5 decimal places
    batch_size = obs.shape[0]
    offset = torch.tensor([-0.615, 0, 0], device=device)

    # ===============================
    # Split observation into components
    # ===============================
    joint_positions = obs[:, :9] + torch.normal(0, noise, size=obs[:, :9].shape, device=device)
    joint_velocities = obs[:, 9:18]+ torch.normal(0, noise, size=obs[:, 9:18].shape, device=device)
    is_grasped = obs[:, 18:19] 
    tcp_pose = obs[:, 19:26] + torch.normal(0, noise, size=obs[:, 19:26].shape, device=device)
    goal_position = obs[:, 26:29] 
    obj_pose = obs[:, 29:36] 
    tcp_to_obj_pos = obs[:, 36:39]
    obj_to_goal_pos = obs[:, 39:42]

    # ===============================
    # Compute forward kinematics for the batch
    # ===============================
    joint_se3_pos, joint_se3_rot = compute_fk(joint_positions[:,:7], end_only=False)
    joint_se3_pos += offset

    # Take the first 7 joints for position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8]  # Shape: (batch_size, 7, 3)
    joint_se3_rot = joint_se3_rot[:, 1:8]  # Shape: (batch_size, 7, 4)
    joint_se3_pose = torch.cat([joint_se3_pos, joint_se3_rot], dim=-1)  # Shape: (batch_size, 7, 7)

    # ===============================
    # Prepare node features per node type
    # ===============================
    data = HeteroData()

    # --- Joint Nodes ---
    num_joints = 7
    joint_node_features = joint_se3_pose.view(-1, 7)  # Shape: (batch_size * 7, 7)
    data['joint'].x = joint_node_features
    data['joint'].batch = torch.arange(batch_size, device=device).repeat_interleave(num_joints)

    # --- TCP Node ---
    tcp_node_features = torch.cat([tcp_pose, joint_positions[:,-1].unsqueeze(-1), is_grasped, tcp_to_obj_pos], dim=-1)  # Shape: (batch_size, 12)
    data['tcp'].x = tcp_node_features
    data['tcp'].batch = torch.arange(batch_size, device=device)

    # --- Object Node ---
    data['object'].x = torch.cat([obj_pose, obj_to_goal_pos], dim=-1)  # Shape: (batch_size, 10)
    data['object'].batch = torch.arange(batch_size, device=device)

    # --- Goal Node ---
    data['goal'].x = goal_position  # Shape: (batch_size, 3)
    data['goal'].batch = torch.arange(batch_size, device=device)

    # ===============================
    # Prepare edge indices and attributes per edge type
    # ===============================

    # --- Kinematic Edges (Joint to Joint) ---
    joint_indices = torch.arange(num_joints, device=device)
    edge_index = torch.tensor([[src.item(), dest.item()] for src in joint_indices for dest in joint_indices], device=device).T

    # Adjust edge indices for batching
    batch_edge_offset = num_joints * torch.arange(batch_size, device=device).unsqueeze(1)
    batch_edge_index = edge_index.unsqueeze(0) + batch_edge_offset.unsqueeze(2)  
    # batch_edge_index = batch_edge_index.permute(0,2,1).reshape(2, -1)
    batch_edge_index = batch_edge_index.permute(1,0,2).reshape(2, -1)
    data['joint', 'kinematic', 'joint'].edge_index = batch_edge_index

    # Edge attributes
    src_pos = data['joint'].x[batch_edge_index[0], :3]
    dst_pos = data['joint'].x[batch_edge_index[1], :3]
    data['joint', 'kinematic', 'joint'].edge_attr = compute_edge_attr(src_pos, dst_pos)

    # --- Joint to TCP Edges ---
    joint_to_tcp_src = num_joints - 1 + num_joints * torch.arange(batch_size, device=device)
    joint_to_tcp_dst = torch.arange(batch_size, device=device)
    edge_index = torch.stack([joint_to_tcp_src, joint_to_tcp_dst], dim=0)
    data['joint', 'kinematic', 'tcp'].edge_index = edge_index

    # Edge attributes
    src_pos = data['joint'].x[edge_index[0], :3]
    dst_pos = data['tcp'].x[edge_index[1], :3]
    data['joint', 'kinematic', 'tcp'].edge_attr = compute_edge_attr(src_pos, dst_pos)

    # --- TCP to Object Edges (if grasped) ---
    grasped_mask = is_grasped.squeeze(-1).bool()
    if grasped_mask.any():
        tcp_indices = torch.arange(batch_size, device=device)[grasped_mask]
        object_indices = tcp_indices  # Assuming one object per batch element

        edge_index = torch.stack([tcp_indices, object_indices], dim=0)
        data['tcp', 'kinematic', 'object'].edge_index = edge_index

        # Edge attributes
        src_pos = data['tcp'].x[edge_index[0], :3]
        dst_pos = data['object'].x[edge_index[1], :3]
        data['tcp', 'kinematic', 'object'].edge_attr = compute_edge_attr(src_pos, dst_pos)

    # --- Interaction Edges ---
    # Simplify by connecting TCP to Object and Goal
    edge_index = torch.arange(batch_size, device=device).unsqueeze(0).repeat(2, 1)
    data['tcp', 'interaction', 'object'].edge_index = edge_index
    data['object', 'interaction', 'goal'].edge_index = edge_index
    # data['tcp', 'interaction', 'goal'].edge_index = edge_index

    # Edge attributes
    src_pos = data['tcp'].x[:, :3]
    dst_pos = data['object'].x[:, :3]
    data['tcp', 'interaction', 'object'].edge_attr = compute_edge_attr(src_pos, dst_pos)
    # dst_pos = data['goal'].x
    # data['tcp', 'interaction', 'goal'].edge_attr = compute_edge_attr(src_pos, dst_pos)
    src_pos = data['object'].x[:, :3]
    dst_pos = data['goal'].x
    data['object', 'interaction', 'goal'].edge_attr = compute_edge_attr(src_pos, dst_pos)

    # Joint to TCP Interaction Edges
    
    # joint_indices = torch.arange(num_joints * batch_size, device=device)
    # tcp_indices = torch.arange(batch_size, device=device).repeat_interleave(num_joints)
    # edge_index = torch.stack([joint_indices, tcp_indices], dim=0)
    # data['joint', 'interaction', 'tcp'].edge_index = edge_index

    # # Edge attributes
    # src_pos = data['joint'].x[edge_index[0], :3]
    # dst_pos = data['tcp'].x[edge_index[1], :3]
    # data['joint', 'interaction', 'tcp'].edge_attr = compute_edge_attr(src_pos, dst_pos)


    # --- Proximity Edges ---
    # Optional: Simplify or remove proximity edges if they don't contribute positively
    # For example, you might only connect objects that are within a certain distance
    # You can vectorize this computation as well

    # ===============================
    # Finalize the graph
    # ===============================
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops(attr="edge_attr")(data)

    if verbose:
        print('Graph is valid: ', data.validate())

    # visualize_graph(data)
    return data

def visualize_graph(data):
    # Visualize the graph using NetworkX and pyvis
    # Visualize the different node types and edge types
    g = to_networkx(data, to_multi=True)
    net = Network(directed=True)
    net.from_nx(g)
    # Set node colors based on node type
    for node in net.nodes:
        if node['type'].startswith('joint'):
            node['color'] = 'blue'
        elif node['type'].startswith('tcp'):
            node['color'] = 'green'
        elif node['type'].startswith('object'):
            node['color'] = 'red'
        elif node['type'].startswith('goal'):
            node['color'] = 'yellow'

    # Set edge colors based on edge type
    for edge in net.edges:
        if 'kinematic' in edge['type'][1]:
            edge['color'] = 'blue'
        elif 'interaction' in edge['type'][1]:
            edge['color'] = 'green'

    net.set_edge_smooth('dynamic')

    net.show('graph.html', notebook=False)

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


def create_hetero_push_cube_graph_batched(obs: np.ndarray, verbose=False):
    batch_size = obs.shape[0]

    # The offset of the robot in the scene must be added to the joint positions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert observations to PyTorch tensor
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    # Truncate all values in obs to have maximal 5 decimal places
    batch_size = obs.shape[0]
    offset = torch.tensor([0, 0, 0], device=device)

    # Split observation into components
    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:18]
    tcp_pose = obs[:, 18:25]
    goal_position = obs[:, 25:28]
    obj_pose = obs[:, 28:35]
    tcp_to_obj_pos = obj_pose[:, :3] - tcp_pose[:, :3]
    obj_to_goal_pos = goal_position - obj_pose[:, :3]

    # ===============================
    # Compute forward kinematics for the batch
    # ===============================
    joint_se3_pos, joint_se3_rot = compute_fk(joint_positions[:,:7], end_only=False)
    joint_se3_pos += offset

    # Take the first 7 joints for position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8]  # Shape: (batch_size, 7, 3)
    joint_se3_rot = joint_se3_rot[:, 1:8]  # Shape: (batch_size, 7, 4)
    joint_se3_pose = torch.cat([joint_se3_pos, joint_se3_rot], dim=-1)  # Shape: (batch_size, 7, 7)

    # ===============================
    # Prepare node features per node type
    # ===============================
    data = HeteroData()

    # --- Joint Nodes ---
    num_joints = 7
    joint_node_features = joint_se3_pose.view(-1, 7)  # Shape: (batch_size * 7, 7)
    data['joint'].x = joint_node_features
    data['joint'].batch = torch.arange(batch_size, device=device).repeat_interleave(num_joints)

    # --- TCP Node ---
    tcp_node_features = torch.cat([tcp_pose, joint_positions[:,-1].unsqueeze(-1), tcp_to_obj_pos], dim=-1)  # Shape: (batch_size, 11)
    data['tcp'].x = tcp_node_features
    data['tcp'].batch = torch.arange(batch_size, device=device)

    # --- Object Node ---
    data['object'].x = torch.cat([obj_pose, obj_to_goal_pos], dim=-1)  # Shape: (batch_size, 10)
    data['object'].batch = torch.arange(batch_size, device=device)

    # --- Goal Node ---
    data['goal'].x = goal_position  # Shape: (batch_size, 3)
    data['goal'].batch = torch.arange(batch_size, device=device)

    # ===============================
    # Prepare edge indices and attributes per edge type
    # ===============================

    # --- Kinematic Edges (Joint to Joint) ---
    joint_indices = torch.arange(num_joints, device=device)
    edge_index = torch.tensor([[src.item(), dest.item()] for src in joint_indices for dest in joint_indices], device=device).T

    # Adjust edge indices for batching
    batch_edge_offset = num_joints * torch.arange(batch_size, device=device).unsqueeze(1)
    batch_edge_index = edge_index.unsqueeze(0) + batch_edge_offset.unsqueeze(2)  
    # batch_edge_index = batch_edge_index.permute(0,2,1).reshape(2, -1)
    batch_edge_index = batch_edge_index.permute(1,0,2).reshape(2, -1)
    data['joint', 'kinematic', 'joint'].edge_index = batch_edge_index

    # Edge attributes
    src_pos = data['joint'].x[batch_edge_index[0], :3]
    dst_pos = data['joint'].x[batch_edge_index[1], :3]
    data['joint', 'kinematic', 'joint'].edge_attr = compute_edge_attr(src_pos, dst_pos)

    # --- Joint to TCP Edges ---
    joint_to_tcp_src = num_joints - 1 + num_joints * torch.arange(batch_size, device=device)
    joint_to_tcp_dst = torch.arange(batch_size, device=device)
    edge_index = torch.stack([joint_to_tcp_src, joint_to_tcp_dst], dim=0)
    data['joint', 'kinematic', 'tcp'].edge_index = edge_index

    # Edge attributes
    src_pos = data['joint'].x[edge_index[0], :3]
    dst_pos = data['tcp'].x[edge_index[1], :3]
    data['joint', 'kinematic', 'tcp'].edge_attr = compute_edge_attr(src_pos, dst_pos)


    # --- Interaction Edges ---
    # Simplify by connecting TCP to Object and Goal
    edge_index = torch.arange(batch_size, device=device).unsqueeze(0).repeat(2, 1)
    data['tcp', 'interaction', 'object'].edge_index = edge_index
    data['object', 'interaction', 'goal'].edge_index = edge_index
    # data['tcp', 'interaction', 'goal'].edge_index = edge_index

    # Edge attributes
    src_pos = data['tcp'].x[:, :3]
    dst_pos = data['object'].x[:, :3]
    data['tcp', 'interaction', 'object'].edge_attr = compute_edge_attr(src_pos, dst_pos)
    # dst_pos = data['goal'].x
    # data['tcp', 'interaction', 'goal'].edge_attr = compute_edge_attr(src_pos, dst_pos)
    src_pos = data['object'].x[:, :3]
    dst_pos = data['goal'].x
    data['object', 'interaction', 'goal'].edge_attr = compute_edge_attr(src_pos, dst_pos)

    # Joint to TCP Interaction Edges
    
    # joint_indices = torch.arange(num_joints * batch_size, device=device)
    # tcp_indices = torch.arange(batch_size, device=device).repeat_interleave(num_joints)
    # edge_index = torch.stack([joint_indices, tcp_indices], dim=0)
    # data['joint', 'interaction', 'tcp'].edge_index = edge_index

    # # Edge attributes
    # src_pos = data['joint'].x[edge_index[0], :3]
    # dst_pos = data['tcp'].x[edge_index[1], :3]
    # data['joint', 'interaction', 'tcp'].edge_attr = compute_edge_attr(src_pos, dst_pos)


    # --- Proximity Edges ---
    # Optional: Simplify or remove proximity edges if they don't contribute positively
    # For example, you might only connect objects that are within a certain distance
    # You can vectorize this computation as well

    # ===============================
    # Finalize the graph
    # ===============================
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops(attr="edge_attr")(data)

    if verbose:
        print('Graph is valid: ', data.validate())

    # visualize_graph(data)
    return data


def create_hetero_stack_cube_graph_batched(obs: np.ndarray):
    batch_size = obs.shape[0]

    # The offset of the robot in the scene must be added to the joint positions
    offset = th.tensor([-0.615, 0, 0]).to(device)

    # Split observation into components
    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:18]
    tcp_pose = obs[:, 18:25]
    cubeA_pose = obs[:, 25:32]
    cubeB_pose = obs[:, 32:39]
    tcp_to_cubeA_pos = obs[:, 39:42]
    tcp_to_cubeB_pos = obs[:, 42:45]
    cubeA_to_cubeB_pos = obs[:, 45:48]

    # Compute forward kinematics for the batch
    joint_se3_pos, joint_se3_rot = compute_fk(joint_positions, end_only=False)
    joint_se3_pos = joint_se3_pos + offset
    if joint_se3_pos.dim() == 2:
        joint_se3_pos = joint_se3_pos.unsqueeze(0)
        joint_se3_rot = joint_se3_rot.unsqueeze(0)
    else:
        joint_se3_pos = joint_se3_pos.permute(1, 0, 2)
        joint_se3_rot = joint_se3_rot.permute(1, 0, 2)

    # Take the first 7 joints for SE(3) position and rotation
    joint_se3_pos = joint_se3_pos[:, 1:8].detach().cpu().numpy()
    joint_se3_rot = rot_normalize(joint_se3_rot[:, 1:8].detach().cpu().numpy())
    joint_se3_pose = np.concatenate([joint_se3_pos, joint_se3_rot], axis=-1)

    # Node Features
    # -------------------------------
    # Joint Nodes
    node_type = np.tile([1, 0, 0, 0], (batch_size, 1))
    joint_node = np.concatenate(
        [
            np.repeat(node_type[:, np.newaxis], 7, axis=1),  # (batch_size, 7, 4)
            joint_se3_pose,  # (batch_size, 7, 7)
            joint_positions[:, :7, np.newaxis],  # (batch_size, 7, 1)
            joint_velocities[:, :7, np.newaxis],  # (batch_size, 7, 1)
            np.tile(np.zeros(13), (batch_size, 7, 1)),  # Padding
        ],
        axis=-1,
    )
    # TCP Node
    node_type = np.tile([0, 1, 0, 0], (batch_size, 1))
    tcp_node = np.concatenate(
        [
            np.repeat(node_type, 1, axis=1),  # (batch_size, 4)
            tcp_pose,  # (batch_size, 7)
            joint_positions[:, 8, np.newaxis],  # (batch_size, 1)
            np.tile(np.zeros(14), (batch_size, 1)),  # Padding
        ],
        axis=-1,
    )
    # Global Node
    node_type = np.tile([0, 0, 1, 0], (batch_size, 1))
    global_node = np.concatenate(
        [
            np.repeat(node_type, 1, axis=1),  # (batch_size, 4)
            np.tile(np.zeros(9), (batch_size, 1)),  # Padding
            tcp_to_cubeA_pos,  # (batch_size, 3)
            cubeA_to_cubeB_pos,  # (batch_size, 3)
            np.tile(np.zeros(7), (batch_size, 1)),  # Padding
        ],
        axis=-1,
    )

    # Cube A Node
    node_type = np.tile([0, 0, 0, 1], (batch_size, 1))
    cubeA_node = np.concatenate(
        [
            np.repeat(node_type, 1, axis=1),  # (batch_size, 4)
            np.tile(np.zeros(15), (batch_size, 1)),  # Padding
            cubeA_pose,  # (batch_size, 7)
        ],
        axis=-1,
    )

    # Cube B Node
    node_type = np.tile([0, 0, 0, 1], (batch_size, 1))
    cubeB_node = np.concatenate(
        [
            np.repeat(node_type, 1, axis=1),  # (batch_size, 4)
            np.tile(np.zeros(15), (batch_size, 1)),  # Padding
            cubeB_pose,  # (batch_size, 7)
        ],
        axis=-1,
    )

    # Edge Features
    # -------------------------------
    # Joint Edges
    joint_joint_edges = [[i, i + 1] for i in range(6)]
    joint_joint_edges = joint_joint_edges + [[i + 1, i] for i in range(6)]
    joint_joint_edges = np.array(joint_joint_edges)
    dist = np.diff(joint_se3_pose[:, :, :3], axis=1)
    rev_dist = -1 * dist
    norm = np.linalg.norm(dist, axis=-1)
    joint_joint_edge_attr = np.concatenate([dist, norm[:, :, np.newaxis]], axis=-1)
    joint_joint_edge_attr_rev = np.concatenate(
        [rev_dist, norm[:, :, np.newaxis]], axis=-1
    )
    joint_joint_edge_attr = np.concatenate(
        [joint_joint_edge_attr, joint_joint_edge_attr_rev], axis=1
    )

    # TCP Edges
    tcp_joint_edge = [[7, 6]]
    joint_tcp_edge = [[6, 7]]
    dist = tcp_pose[:, :3] - joint_se3_pose[:, -1, :3]
    norm = np.linalg.norm(dist, axis=-1)
    tcp_joint_edge_attr = np.concatenate([dist, norm[:, np.newaxis]], axis=-1)
    joint_tcp_edge_attr = np.concatenate([dist, norm[:, np.newaxis]], axis=-1)

    # Global Edges
    global_joint_edges = [[8, i] for i in range(7)]
    global_tcp_edges = [[8, 7]]
    global_joint_edge_attr = np.ones((batch_size, 7, 4))
    global_tcp_edge_attr = np.ones((batch_size, 1, 4))

    # Cube A Edges
    cubeA_global_edges = [[9, 8]]
    cubeA_cubeB_edges = [[9, 10]]
    cubeA_global_edge_attr = np.ones((batch_size, 1, 4))
    dist = cubeA_to_cubeB_pos
    norm = np.linalg.norm(dist, axis=-1)
    cubeA_cubeB_edge_attr = np.concatenate([dist, norm[:, np.newaxis]], axis=-1)

    # Cube B Edges
    cubeB_cubeA_edges = [[10, 9]]
    dist = -1 * cubeA_to_cubeB_pos
    norm = np.linalg.norm(dist, axis=-1)
    cubeB_cubeA_edge_attr = np.concatenate([dist, norm[:, np.newaxis]], axis=-1)
    cubeB_global_edges = [[10, 8]]
    cubeB_global_edge_attr = np.ones((batch_size, 1, 4))

    # Create the graph
    # -------------------------------
    graph_list = []
    for batch in range(batch_size):
        data = Data()
        data.x = th.tensor(
            np.concatenate(
                [
                    joint_node[batch],
                    tcp_node[batch][np.newaxis, :],
                    global_node[batch][np.newaxis, :],
                    cubeA_node[batch][np.newaxis, :],
                    cubeB_node[batch][np.newaxis, :],
                ],
            )
        ).float()

        # Edge Index
        data.edge_index = (
            th.tensor(
                np.concatenate(
                    [
                        joint_joint_edges,
                        tcp_joint_edge,
                        joint_tcp_edge,
                        global_joint_edges,
                        global_tcp_edges,
                        cubeA_global_edges,
                        cubeA_cubeB_edges,
                        cubeB_cubeA_edges,
                        cubeB_global_edges,
                    ],
                    axis=0,
                )
            )
            .long()
            .T
        )

        # Edge Attributes
        data.edge_attr = th.tensor(
            np.concatenate(
                [
                    joint_joint_edge_attr[batch],
                    tcp_joint_edge_attr[batch][np.newaxis, :],
                    joint_tcp_edge_attr[batch][np.newaxis, :],
                    global_joint_edge_attr[batch],
                    global_tcp_edge_attr[batch],
                    cubeA_global_edge_attr[batch],
                    cubeA_cubeB_edge_attr[batch][np.newaxis, :],
                    cubeB_cubeA_edge_attr[batch][np.newaxis, :],
                    cubeB_global_edge_attr[batch],
                ],
                axis=0,
            )
        ).float()

        data = T.AddSelfLoops(attr="edge_attr", fill_value=0)(data)
        data.cpu()
        graph_list.append(data)

        # x = to_networkx(data)
        # nx.draw(x, with_labels=True)
        # plt.show()

    return graph_list
