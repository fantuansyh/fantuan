import os
import torch
import numpy as np
from torch_geometric.data import Data
import esm
import pickle

# 加载预训练的 ESM 模型
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model.eval()

# 从 PDB 文件中提取 ESM 特征的函数
def parse_pdb_and_get_esm_features(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    coords = []
    chain_sequence = []
    residue_indices = []
    current_residue_index = None

    for line in lines:
        if line.startswith('ATOM') and line[13:15].strip() == 'CA':  # 仅考虑 α 碳
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            residue_index = int(line[22:26].strip())
            residue_type = line[17:20].strip()

            if residue_index != current_residue_index:
                coords.append((x, y, z))
                residue_indices.append(residue_index)
                chain_sequence.append(residue_type)
                current_residue_index = residue_index

    aa_mapping = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    sequence = ''.join([aa_mapping.get(res, 'X') for res in chain_sequence])

    token_ids = torch.tensor([alphabet.encode(sequence)])
    with torch.no_grad():
        results = model(token_ids, repr_layers=[33])
    features = results['representations'][33].squeeze(0)

    return coords, features, residue_indices

def build_graph_from_esm_features(filepath, threshold=6.0):
    coords, node_features, residue_indices = parse_pdb_and_get_esm_features(filepath)
    num_residues = len(residue_indices)

    edges = []
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if np.linalg.norm(np.array(coords[i]) - np.array(coords[j])) < threshold:
                edges.append([i, j])
                edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=node_features, edge_index=edge_index)
    return data

def create_graph_data_list(root_dir, subdirs, save_path):
    graph_data_list = []

    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"Directory not found: {subdir_path}, skipping...")
            continue

        pdb_files = [f for f in os.listdir(subdir_path) if f.endswith('.pdb')]
        for pdb_file in pdb_files:
            pdb_filepath = os.path.join(subdir_path, pdb_file)
            graph_data = build_graph_from_esm_features(pdb_filepath)
            if 'label_0' in subdir:
                graph_data.y = torch.tensor([0])
            elif 'label_1' in subdir:
                graph_data.y = torch.tensor([1])
            elif 'label_2' in subdir:
                graph_data.y = torch.tensor([2])

            graph_data_list.append(graph_data)

    # Save the processed data to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(graph_data_list, f)
    print(f"Graph data has been saved to {save_path}")

    return graph_data_list

def load_or_create_graph_data_list(root_dir, subdirs, save_path):
    if os.path.exists(save_path):
        print(f"Loading processed graph data from {save_path}")
        with open(save_path, 'rb') as f:
            graph_data_list = pickle.load(f)
    else:
        print(f"No processed graph data found at {save_path}, processing new data...")
        graph_data_list = create_graph_data_list(root_dir, subdirs, save_path)

    return graph_data_list

# 使用创建的函数
# root_dir = 'E:/iGEM/前期准备/GCN/data/24.4.18 pdb_3D_dataset'
# subdirs = ['label_0', 'label_1', 'label_2']
# save_path = 'E:/iGEM/前期准备/GCN/data/processed_graph_data.pkl'  # Define the path to save the processed data
# graph_data_list = load_or_create_graph_data_list(root_dir, subdirs, save_path)
# print(f'Total graphs loaded: {len(graph_data_list)}')
