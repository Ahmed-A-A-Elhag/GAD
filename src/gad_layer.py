############ GAD layer #####################
import torch
import torch.nn as nn

from aggregators import AGGREGATORS
from mlp import MLP
from scalers import SCALERS
from dgn_layer import DGN_layer_Simple, DGN_Tower, DGN_layer_Tower
from diffusion_layer import Diffusion_layer

class GAD_layer(nn.Module):
    def __init__(self, hid_dim, graph_norm, batch_norm, dropout, aggregators, scalers, edge_fts, avg_d, D, device, towers, type_net, residual, use_diffusion, diffusion_method, k):
        super().__init__()
        
        aggregators = [aggr for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]
        
        if type_net == 'simple':
            self.DGN_layer = DGN_layer_Simple(hid_dim = hid_dim, graph_norm = graph_norm, batch_norm = batch_norm, aggregators = aggregators,
                                              scalers = scalers, edge_fts = edge_fts, avg_d = avg_d, D = D, device = device)
        elif type_net == 'tower':
            self.DGN_layer = DGN_layer_Tower(hid_dim = hid_dim, graph_norm = graph_norm, batch_norm = batch_norm, aggregators = aggregators,
                                              scalers = scalers, edge_fts = edge_fts, avg_d = avg_d, D = D, device = device, towers=towers)
            
        self.dropout = dropout  
        self.use_diffusion = use_diffusion
        if self.use_diffusion:
            self.diffusion_layer = Diffusion_layer(hid_dim, method = diffusion_method, k = k, device = device)
            self.MLP_last = MLP([2*hid_dim, hid_dim], dropout = False)
            
        self.residual = residual

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx):

        if self.use_diffusion:
  
            diffusion_out  =  self.diffusion_layer(node_fts, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, batch_idx) 
            dgn_out         = self.DGN_layer(diffusion_out, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n)
            output          = torch.cat((diffusion_out, dgn_out), dim=1)
            output          = self.MLP_last(output)
        else:
            output          = self.DGN_layer(node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, norm_n)
            

        if self.residual:
            output   = node_fts + output
            
#         output = nn.functional.dropout(output, self.dropout, training=self.training)

        return output
