## Description 
Here we include the scripts to generate our results on QM9 dataset, including uploading the data, preprocessing steps, etc. 

***To run GAD model on QM9 properties***:

Dipole moment <img src="https://render.githubusercontent.com/render/math?math=\mu"> (Unit: D)

```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=0 --factor=1 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

Isotropic polarizability <img src="https://render.githubusercontent.com/render/math?math=\alpha">  
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=1 --factor=1 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

Highest occupied molecular orbital energy <img src="https://render.githubusercontent.com/render/math?math=\epsilon">  
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=2 --factor=1000 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```
Gap between <img src="https://render.githubusercontent.com/render/math?math=\epsilon_{HOMO}">  and <img src="https://render.githubusercontent.com/render/math?math=\epsilon_{LUMO}">  <img src="https://render.githubusercontent.com/render/math?math=\Delta \epsilon`"> 
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=3 --factor=1000 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

Highest occupied molecular orbital energy <img src="https://render.githubusercontent.com/render/math?math=\epsilon_{\textrm{HOMO}">  
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=4 --factor=1000 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

\epsilon_{\textrm{HOMO}}
