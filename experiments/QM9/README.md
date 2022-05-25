## Description 
Here we include the scripts to generate our results on QM9 dataset, including uploading the data, preprocessing steps, etc. 

***To run GAD model on QM9 properties***:

Dipole moment <img src="https://render.githubusercontent.com/render/math?math=\mu"> (Unit: D)

```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=0 --factor=1 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```

Isotropic polarizability <img src="https://render.githubusercontent.com/render/math?math=\alpha"> (Unit: <img src="https://render.githubusercontent.com/render/math?math=\mu">)
```
python -m main_QM9 --n_layers=8 --hid_dim=100 --atomic_emb=50 --dropout=0 --readout='mean' --use_diffusion=True --diffusion_method='spectral' --k=25 --aggregators='mean sum max dir_der' --scalers='identity amplification attenuation' --use_edge_fts=True --use_graph_norm=True --use_batch_norm=True --use_residual=True --type_net='tower' --towers=5 --prop_idx=1 --factor=1 --num_epochs=256 --batch_size=256 --lr=1e-3 --weight_decay=3e-6 --min_lr=1e-5
```
