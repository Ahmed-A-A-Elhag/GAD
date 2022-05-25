############ GAD model #####################
import torch
import torch.nn as nn

from torch_scatter import scatter

from aggregators import AGGREGATORS
from mlp import MLP
from scalers import SCALERS
from dgn_layer import DGN_layer_Simple, DGN_Tower, DGN_layer_Tower

from gad_layer import GAD_layer

class GAD(nn.Module):
    def __init__(self, num_of_node_fts, num_of_edge_fts, hid_dim, atomic_emb, graph_norm, batch_norm, dropout, readout, aggregators, scalers, edge_fts, avg_d, D, device, towers, type_net, residual, use_diffusion, diffusion_method, k, n_layers):
        super().__init__()
        
        self.hidden_dim    = hid_dim
        self.atomic_emb    = atomic_emb
        self.n_layers      = n_layers
        self.type_net      = type_net

        self.readout       = readout
        self.graph_norm    = graph_norm
        self.batch_norm    = batch_norm
        self.aggregators   = aggregators
        self.scalers       = scalers
        self.avg_d         = avg_d
        self.residual      = residual
        self.edge_fts      = edge_fts
        self.device        = device
        
        if self.type_net == 'simple':
            self.edge_dim = self.hidden_dim 
        elif self.type_net == 'tower':
            self.edge_dim = (self.hidden_dim + self.atomic_emb) // towers
   
        if self.edge_fts:
            self.layer_first_edge = nn.Linear(num_of_edge_fts, (self.hidden_dim + self.atomic_emb)//towers)
            
        self.emb_automic = nn.Embedding(10, self.atomic_emb)
    
        self.layer_first_node = nn.Linear(num_of_node_fts, self.hidden_dim)
        
        self.layer_first = nn.Linear(self.hidden_dim + self.atomic_emb, self.hidden_dim + self.atomic_emb)
        self.layer_last  = nn.Linear(self.hidden_dim + self.atomic_emb, self.hidden_dim + self.atomic_emb)
        
        
        self.layers = nn.ModuleList([GAD_layer(hid_dim = self.hidden_dim + self.atomic_emb, graph_norm = self.graph_norm, batch_norm = self.batch_norm, 
                                    dropout = dropout, aggregators = self.aggregators, scalers = self.scalers, edge_fts = self.edge_fts, 
                                    avg_d = self.avg_d, D = D, device= self.device, towers=towers, type_net = self.type_net, 
                                    residual = self.residual, use_diffusion = use_diffusion, 
                                                        diffusion_method = diffusion_method, k =k) for _ in range(self.n_layers)])
        

        self.readout_MLP = MLP([ self.hidden_dim + self.atomic_emb, (self.hidden_dim + self.atomic_emb)//2,  1], dropout = False)



    def forward(self, node_fts, automic_num, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx):
      

        if self.edge_fts:
            edge_fts = self.layer_first_edge(edge_fts)

        automic_num = self.emb_automic(automic_num)
        node_fts = self.layer_first_node(node_fts)
        
        node_fts = torch.cat((node_fts, automic_num), dim = 1)
        node_fts = self.layer_first(node_fts)
        
        
        for i, conv in enumerate(self.layers):
            new_node_fts = conv(node_fts, edge_fts, edge_index, F_norm_edge, F_dig, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, norm_n, batch_idx)
            node_fts = new_node_fts

            
        output  = self.layer_last(node_fts)

        output = scatter(output, batch_idx, dim=0, reduce= self.readout)

        output = self.readout_MLP(output)
        
        output = output.squeeze(-1)


        return output
