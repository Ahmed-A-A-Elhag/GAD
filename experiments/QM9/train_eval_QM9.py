import torch
import torch.nn as nn

from tqdm import tqdm

from torch_geometric.utils import to_dense_adj, get_laplacian

def evaluate_network(model, data_loader, prop_idx, factor, device):

    model.eval()
    
    epoch_test_mae = 0


    loss_fn = nn.L1Loss()

    with torch.no_grad():

        for idx, batched_graph in tqdm(enumerate(data_loader)):

            
            
            num_nodes = batched_graph.num_nodes
            adj = to_dense_adj(batched_graph.edge_index, max_num_nodes = num_nodes)[0].to(device)
            node_deg_vec = adj.sum(axis=1, keepdim=True).to(device)
            
            node_deg_mat = torch.diag(node_deg_vec[:, 0]).to(device)

            lap_mat_sparse = get_laplacian(batched_graph.edge_index)
            lap_mat = to_dense_adj(edge_index = lap_mat_sparse[0], edge_attr = lap_mat_sparse[1], max_num_nodes = num_nodes)[0].to(device)
   
            F_norm_edge = batched_graph.F_norm_edge.to(device)
            F_dig = batched_graph.F_dig.to(device)
            
            edge_index = batched_graph.edge_index.to(device)
            batch_idx = batched_graph.batch.to(device)

            node_fts = batched_graph.x.to(device)
            node_fts = node_fts.squeeze()
            
            automic_num = batched_graph.z.to(device)
            
            edge_fts = batched_graph.edge_attr.to(device)
            norm_n = batched_graph.norm_n.to(device)
            
            k_eig_val = batched_graph.k_eig_val.to(device)
            k_eig_vec = batched_graph.k_eig_vec.to(device)
     
            out_model = model(node_fts, automic_num, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)
        
            loss = loss_fn(out_model, batched_graph.y[:, prop_idx].to(device))

            epoch_test_mae += loss.item()  # MAE: mean absolute error


        epoch_test_mae /= (idx + 1)
        
    return factor*epoch_test_mae


def train_epoch(model ,data_loader, optimizer, prop_idx, factor, device, loss_fn):
    
        epoch_train_mae = 0

        model.train() 

        for idx, batched_graph in tqdm(enumerate(data_loader)):

            optimizer.zero_grad()
        
            num_nodes = batched_graph.num_nodes
            
            adj = to_dense_adj(batched_graph.edge_index, max_num_nodes = num_nodes)[0].to(device)
            node_deg_vec = adj.sum(axis=1, keepdim=True).to(device)
            
            node_deg_mat = torch.diag(node_deg_vec[:, 0]).to(device)

            
            lap_mat_sparse = get_laplacian(batched_graph.edge_index)
            lap_mat = to_dense_adj(edge_index = lap_mat_sparse[0], edge_attr = lap_mat_sparse[1], max_num_nodes = num_nodes)[0].to(device)

        
            F_norm_edge = batched_graph.F_norm_edge.to(device)
            F_dig = batched_graph.F_dig.to(device)
     
            edge_index = batched_graph.edge_index.to(device)
            batch_idx = batched_graph.batch.to(device)

            node_fts = batched_graph.x.to(device)
            node_fts = node_fts.squeeze()
            
            automic_num = batched_graph.z.to(device)
            
            edge_fts = batched_graph.edge_attr.to(device)
            
            norm_n = batched_graph.norm_n.to(device)
            
            k_eig_val = batched_graph.k_eig_val.to(device)
            k_eig_vec = batched_graph.k_eig_vec.to(device)

            out_model = model(node_fts, automic_num, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)

            loss = loss_fn(out_model, batched_graph.y[:, prop_idx].to(device))
            
            loss.backward()
            optimizer.step()

            epoch_train_mae += loss.item() 

        epoch_train_mae /= (idx + 1)
        return factor*epoch_train_mae, optimizer
