import torch
import torch.optim as opt
import torch.nn as nn

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
import os
import sys

from tqdm import tqdm

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/")) 

from preprocessing import preprocessing_dataset, average_node_degree
from evaluate import train_epoch, evaluate_network
from GAD import GAD

def train_ZINC(model, optimizer, train_loader, val_loader, device, num_epochs):

    loss_fn = nn.L1Loss()

    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, 
                                                     patience=15,
                                                   threshold=0.004,
                                                   verbose=True)

    epoch_train_MAEs, epoch_val_MAEs = [], []
    
    Best_val_mae = 10

    for epoch in range(num_epochs):
        
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("\n!! LR EQUAL TO MIN LR SET.")
            break
        
        epoch_train_mae, optimizer = train_epoch(model ,train_loader, optimizer, device, loss_fn)
        epoch_val_mae = evaluate_network(model,  val_loader, device)

        epoch_train_MAEs.append(epoch_train_mae)
        epoch_val_MAEs.append(epoch_val_mae)

        scheduler.step(epoch_val_mae)
        if(epoch_val_mae < Best_val_mae):
            Best_val_mae =  epoch_val_mae
            torch.save(model, 'model.pth')

        torch.save(model, 'model_running.pth')

        print("")
        print("epoch_idx", epoch)
        print("epoch_train_MAEs", epoch_train_mae)
        print("epoch_val_MAEs", epoch_val_mae)
        
        
def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--graph_norm', help="Enter true for graph_norm", type = bool ,default=True)
    parser.add_argument('--batch_norm', help="Enter true for batch_norm", type = bool ,default=True)
    parser.add_argument('--L', help="Enter the number of layers", type = int)
    parser.add_argument('--dropout', help="Enter the rate of the dropout", type = int, default=0)
    parser.add_argument('--readout', help="Enter the readout agggregator", type = str, default='mean')
    parser.add_argument('--aggregators', help="Enter the aggregators", type = str)
    parser.add_argument('--scalers', help="Enter the scalers", type = str)
    parser.add_argument('--edge_fts', help="Enter true if you want to use edge_fts", type = bool)
    parser.add_argument('--type_net', help="Enter the type_net", type = str)
    parser.add_argument('--towers', help="Enter the num of towers", type = int)
    parser.add_argument('--residual', help="Enter the residual", type = bool)
    parser.add_argument('--use_diffusion', help="Enter the use_diffusion", type = bool)
    parser.add_argument('--diffusion_method', help="Enter the diffusion_method", type = str)
    parser.add_argument('--k', help="Enter the num k", type = int)
    
    args = parser.parse_args()
    
    dataset_train = ZINC(root='/', subset=True)
    dataset_val = ZINC(root='/', subset=True, split='val')
    dataset_test = ZINC(root='/', subset=True, split='test')
    print("bn")

    D, avg_d = average_node_degree(dataset_train)
    dataset_train = preprocessing_dataset(dataset_train, 30)
    dataset_val = preprocessing_dataset(dataset_val, 30)
    dataset_test = preprocessing_dataset(dataset_test, 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(dataset = dataset_train, batch_size=16,shuffle=True) 
    val_loader = DataLoader(dataset = dataset_val, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset =  dataset_test, batch_size=16, shuffle=False)
    
    model = GAD(num_atom_type = 28, num_bond_type = 4, hid_dim = 65, graph_norm = True, batch_norm = True, dropout = 0,
                      readout = 'mean', aggregators = 'mean dir_der max min', scalers = 'identity amplification attenuation', 
                      edge_fts = True, avg_d = avg_d, D = D, device = device, towers=5, type_net = 'tower', 
                      residual = True, use_diffusion = True, diffusion_method = 'spectral',k = 30, n_layers = 4)
    
    lr= 1e-3

    
    optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=3e-6)
    
    train_ZINC(model, optimizer, train_loader, val_loader, device, num_epochs = 300)
    
    
main()
    
