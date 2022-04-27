


def get_mask(k, batch_num_nodes, num_nodes, device):
    mask = torch.zeros(num_nodes, k*len(batch_num_nodes)).to(device)
    partial_n = 0
    partial_k = 0
    for n in batch_num_nodes:
        mask[partial_n: partial_n + n, partial_k: partial_k + k] = 1
        partial_n = partial_n + n
        partial_k = partial_k + k
    return mask
  
  
class Diffusion_layer(nn.Module):

    def __init__(self, width, method, k, device):
        super().__init__()

        self.width = width
        self.method = method
        self.k = k
        self.device = device
        self.relu = nn.LeakyReLU()

        self.diffusion_time = nn.Parameter(torch.Tensor(self.width)) # num_channels

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, node_fts, node_deg_vec, node_deg_mat, lap_mat, k_eig_val, k_eig_vec, num_nodes, batch_idx): 
    

        with torch.no_grad():

            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if  self.method == 'spectral':

            _, indices_of_each_graph = torch.unique(batch_idx, return_counts = True)
            
            indices_of_each_graph = indices_of_each_graph.to(self.device)

            batch_size = indices_of_each_graph.shape[0]

            mask = get_mask(self.k, indices_of_each_graph.tolist(), num_nodes, self.device)
            
            k_eig_vec_ = k_eig_vec.repeat(1, batch_size)

            k_eig_vec_ = k_eig_vec_ * mask

            basisT = k_eig_vec_.transpose(-2, -1)

            x_spec = torch.matmul(basisT, node_fts * node_deg_vec)

            time = self.diffusion_time

            diffusion_coefs = torch.exp(-k_eig_val.unsqueeze(-1) * time.unsqueeze(0))

            x_diffuse_spec = diffusion_coefs * x_spec

            x_diffuse      = torch.matmul(k_eig_vec_, x_diffuse_spec)


        elif self.method == 'implicit':
                


            mat_ = lap_mat.unsqueeze(0).expand( self.width, num_nodes, num_nodes).clone()

            mat_ *= self.diffusion_time.unsqueeze(-1).unsqueeze(-1)

            #  torch.diag(mass[:, 0])

            mat_ += node_deg_mat.unsqueeze(0)
            # mat_ =  node_deg_mat.unsqueeze(0) - mat_


            cholesky_factors = torch.linalg.cholesky(mat_)

            # Solve the system
            rhs = node_fts * node_deg_vec

            rhsT = torch.transpose(rhs, 0, 1).unsqueeze(-1)


            sols = torch.cholesky_solve(rhsT, cholesky_factors)

            x_diffuse = torch.transpose(sols.squeeze(-1), 0, 1)

        

        x_diffuse = self.relu(x_diffuse)

        return x_diffuse
