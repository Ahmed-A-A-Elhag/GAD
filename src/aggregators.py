import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing


## mean aggregator

class aggregate_mean(MessagePassing):
    def __init__(self, edge_fts, hid_dim, device):
        super().__init__(aggr='mean') 
        self.device = device

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig):
        # node_fts has shape [num_of_nodes, in_channels]
        # edge_index has shape [2, num_of_edges]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=node_fts.shape[0])
        edge_index = edge_index.to(self.device)
        
        row, _ = edge_index
  
        norm = torch.ones(row.shape[0])
        norm = norm.to(self.device)

        return self.propagate(edge_index, x=node_fts, norm=norm)

    def message(self, x_j, norm):
        
        # x_j has shape [num_of_edges, out_channels]

        return norm.view(-1, 1) * x_j
   


 ## sum aggregator

class aggregate_sum(MessagePassing):
    def __init__(self, edge_fts, hid_dim, device):
        super().__init__(aggr='add') 
        self.device = device

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig):
        # node_fts has shape [num_of_nodes, in_channels]
        # edge_index has shape [2, num_of_edges]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=node_fts.shape[0])
        edge_index = edge_index.to(self.device)
        
        row, _ = edge_index
  
        norm = torch.ones(row.shape[0])
        norm = norm.to(self.device)

        return self.propagate(edge_index, x=node_fts, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [num_of_edges, out_channels]

        return norm.view(-1, 1) * x_j
    
  
      
## max aggregator

class aggregate_max(MessagePassing):
    def __init__(self, edge_fts, hid_dim, device):
        super().__init__(aggr='max') 
        self.device = device

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig):
        # node_fts has shape [num_of_nodes, in_channels]
        # edge_index has shape [2, num_of_edges]

        edge_index, _ = add_self_loops(edge_index, num_nodes=node_fts.shape[0])
        edge_index = edge_index.to(self.device)
        
        row, _ = edge_index

        norm = torch.ones(row.shape[0])
        norm = norm.to(self.device)

        return self.propagate(edge_index, x=node_fts, norm=norm)

    def message(self, x_j, norm):


        return norm.view(-1, 1) * x_j 

      
      
## min aggregator (using max applied to -features)

class aggregate_min(MessagePassing):
    def __init__(self, edge_fts, hid_dim, device):
        super().__init__(aggr='max') 
        self.device = device

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig):
        # node_fts has shape [num_of_nodes, in_channels]
        # edge_index has shape [2, num_of_edges]

        edge_index, _ = add_self_loops(edge_index, num_nodes=node_fts.shape[0])
        edge_index = edge_index.to(self.device)
        
        row, _ = edge_index

        norm = torch.ones(row.shape[0])
        norm = norm.to(self.device)

        out =  self.propagate(edge_index, x= -node_fts, norm=norm)
        return -out

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j
      
      
      
## B_av aggregator 

class aggregate_dir_smooth(MessagePassing):
    def __init__(self, edge_fts, hid_dim, device):
        super().__init__(aggr='add')  
        self.edge_fts = edge_fts
        self.device = device
        if self.edge_fts:
            self.input_dim = 3*hid_dim
        else:
            self.input_dim = 2*hid_dim
        self.output_dim = hid_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim).to(self.device)

    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig):

        return self.propagate(edge_index, x=node_fts, edge_attr=edge_fts, norm=F_norm_edge)

    def message(self, x_i, x_j, edge_attr, norm):
        
        if self.edge_fts:
            new_fts = torch.cat([x_i, abs(norm).view(-1, 1) * x_j, abs(norm).view(-1, 1) * edge_attr], dim = 1)
        else:
            new_fts = torch.cat([x_i, abs(norm).view(-1, 1) * x_j], dim = 1)
        return self.linear(new_fts)
      
   
  
      
## B_dx aggregator 

class aggregate_dir_der(MessagePassing):
    def __init__(self, edge_fts, hid_dim, device):
        super().__init__(aggr='add')  
        
        self.edge_fts = edge_fts
        self.device = device
        if self.edge_fts:
            self.input_dim = 3*hid_dim
        else:
            self.input_dim = 2*hid_dim
        self.output_dim = hid_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim).to(self.device)
        
    def forward(self, node_fts, edge_fts, edge_index, F_norm_edge, F_dig):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes= node_fts.size(0))
        edge_index = edge_index.to(self.device)
        
        if self.edge_fts:
            zero_tensor = torch.zeros((edge_index.shape[1] - edge_fts.shape[0], self.output_dim)).to(self.device)
            new_edge_fts  = torch.cat([edge_fts, zero_tensor])
            return self.propagate(edge_index, x=node_fts, edge_attr=new_edge_fts, norm= (F_norm_edge, F_dig))
        else:
            return self.propagate(edge_index, x=node_fts, edge_attr=edge_fts, norm= (F_norm_edge, F_dig))

    def message(self, x_i, x_j, edge_attr, norm):
        
        norm_1, norm_2 = norm
        norm_3 = torch.cat([norm_1, -norm_2], dim = 0)
        message_1 = abs(norm_3.view(-1, 1) * x_j)
        if self.edge_fts:
            message_2 = abs(norm_3.view(-1, 1) * edge_attr)
            out = torch.cat([x_i, message_1, message_2], dim = 1)
        else:
            out = torch.cat([x_i, message_1], dim = 1)
 
        return self.linear(out)
 
AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min, 'dir_der':aggregate_dir_der, 'dir_smooth':aggregate_dir_smooth}
