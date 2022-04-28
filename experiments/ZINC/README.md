## Description 
Here we include the scripts to generate our results on ZINC dataset, including uploading the data, preprocessing steps, etc. 

***To run GAD model on ZINC***:

GAD using implicit timestep solving scheme
```
python -m main_ZINC --n_layers=4 --hid_dim=65 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='implicit' --k=30 --aggregators='mean dir_der max min' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --num_epochs=300 --batch_size=16 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

GAD using spectral expansion scheme
```
python -m main_ZINC --n_layers=4 --hid_dim=65 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=30 --aggregators='mean dir_der max min' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --num_epochs=300 --batch_size=64 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```
