import torch
import torch.optim as opt
import torch.nn as nn

from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
import os
import sys

from tqdm import tqdm

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../")) 
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/")) 

from ZINC.preprocessing import preprocessing_dataset, average_node_degree
from train_eval_QM9 import train_epoch, evaluate_network
from GAD_QM9.gad import GAD

def train_QM9(model, optimizer, train_loader, val_loader, prop_idx, factor, device, num_epochs, min_lr):

    loss_fn = nn.L1Loss()

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, 
                                                     patience=15,
                                                   threshold=0.004,
                                                   verbose=True)

    epoch_train_MAEs, epoch_val_MAEs = [], []
    
    Best_val_mae = 1000

    print("Start training")

    for epoch in range(num_epochs):
        
        if optimizer.param_groups[0]['lr'] < min_lr:
            print("lr equal to min_lr: exist")
            break
        
        epoch_train_mae, optimizer = train_epoch(model ,train_loader, optimizer, prop_idx, factor, device, loss_fn)
        epoch_val_mae = evaluate_network(model, val_loader, prop_idx, factor, device)

        epoch_train_MAEs.append(epoch_train_mae)
        epoch_val_MAEs.append(epoch_val_mae)

        scheduler.step(epoch_val_mae)
        if(epoch_val_mae < Best_val_mae):
            Best_val_mae =  epoch_val_mae
            torch.save(model, 'model.pth')

        torch.save(model, 'model_running.pth')

        print("")
        print("epoch_idx", epoch)
        print("epoch_train_MAE", epoch_train_mae)
        print("epoch_val_MAE", epoch_val_mae)
        
    print("Finish training")

def main():

    parser = argparse.ArgumentParser()
    

    parser.add_argument('--n_layers', help="Enter the number of GAD layers", type = int)
    parser.add_argument('--hid_dim', help="Enter the hidden dimensions", type = int)
    parser.add_argument('--atomic_emb', help="Enter the embedding dimensions of the atomic number", type = int)
    
    parser.add_argument('--dropout', help="Enter the value of the dropout", type = int, default=0)
    parser.add_argument('--readout', help="Enter the readout agggregator", type = str, default='mean')

    
    parser.add_argument('--use_diffusion', help="Enter the use_diffusion", type = bool)
    parser.add_argument('--diffusion_method', help="Enter the diffusion layer solving scheme ", type = str)
    parser.add_argument('--k', help="Enter the num of eigenvector for spectral scheme", type = int)

    parser.add_argument('--aggregators', help="Enter the aggregators", type = str)
    parser.add_argument('--scalers', help="Enter the scalers", type = str)

    parser.add_argument('--use_edge_fts', help="Enter true if you want to use the edge_fts", type = bool)
    parser.add_argument('--use_graph_norm', help="Enter true if you want to use graph_norm", type = bool ,default=True)
    parser.add_argument('--use_batch_norm', help="Enter true if you want to use batch_norm", type = bool ,default=True)
    parser.add_argument('--use_residual', help="Enter true if you want to use residual connection", type = bool)

    parser.add_argument('--type_net', help="Enter the type_net for DGN layer", type = str)
    parser.add_argument('--towers', help="Enter the num of towers for DGN_tower", type=int)

    parser.add_argument('--prop_idx', help="Enter the QM9 property index", type = int)
    parser.add_argument('--factor', help="Enter the factor 1000 to convert the QM9 property with Unit eV to meV. Enter 1 for the others properties", type = int)

    
    parser.add_argument('--num_epochs', help="Enter the num of epochs", type = int)
    parser.add_argument('--batch_size', help="Enter the batch size", type = int)
    parser.add_argument('--lr', help="Enter the learning rate", type = float)
    parser.add_argument('--weight_decay', help="Enter the weight_decay", type = float)
    parser.add_argument('--min_lr', help="Enter the minimum lr", type = float)
    
    args = parser.parse_args()
    
    print("downloading the dataset (QM9)")
    dataset = QM9(root='/')
    dataset = dataset.shuffle()
    
    dataset_test = dataset[:10000]
    dataset_val = dataset[10000:20000]
    dataset_train = dataset[20000:]
    print("dataset_train contains ", len(dataset_train), "samples")
    print("dataset_val contains ", len(dataset_val), "samples")
    print("dataset_test contains ", len(dataset_test), "samples")

    print("data preprocessing: calculate and store the vector field F, etc.")

    D, avg_d = average_node_degree(dataset_train)
    dataset_train = preprocessing_dataset(dataset_train, args.k)
    dataset_val = preprocessing_dataset(dataset_val, args.k)
    dataset_test = preprocessing_dataset(dataset_test, args.k)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(dataset = dataset_train, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(dataset = dataset_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset =  dataset_test, batch_size=args.batch_size, shuffle=False)

    print("create GAD model")
    
    model = GAD(num_of_node_fts = 11, num_of_edge_fts = 4, hid_dim = args.hid_dim, atomic_emb = args.atomic_emb, graph_norm = args.use_graph_norm, 
               batch_norm = args.use_batch_norm, dropout = args.dropout, readout = args.readout, aggregators = args.aggregators,
               scalers = args.scalers, edge_fts = args.use_edge_fts, avg_d = avg_d, D = D, device = device, towers= args.towers,
               type_net = args.type_net, residual = args.use_residual, use_diffusion = args.use_diffusion, 
               diffusion_method = args.diffusion_method, k = args.k, n_layers = args.n_layers)
    

    model = model.to(device)
    
    optimizer = opt.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_QM9(model, optimizer, train_loader, val_loader, prop_idx = args.prop_idx, factor = args.factor, device=device, num_epochs = args.num_epochs, min_lr = args.min_lr)
    
    print("Uploading the best model")

    model_ = torch.load('model.pth')

    test_mae = evaluate_network(model_, test_loader, args.prop_idx, args.factor, device)
    val_mae = evaluate_network(model_, val_loader, args.prop_idx, args.factor, device)
    train_mae = evaluate_network(model_, train_loader, args.prop_idx, args.factor, device)

    print("")
    print("Best Train MAE: {:.4f}".format(train_mae))
    print("Best Val MAE: {:.4f}".format(val_mae))
    print("Best Test MAE: {:.4f}".format(test_mae))

main()
   
