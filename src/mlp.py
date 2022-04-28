import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, model_dims, dropout = False):
        """
        hid_dim: the dimensions of model layers in a list, including the first layer and the last layer.
        """
        super().__init__()

        self.model_dims = model_dims 

        self.activation = nn.ReLU() 

        self.layers = nn.Sequential() 

        for i in range(len(model_dims)- 1):

            if dropout and i > 0:
                self.add_module(
                    "mlp_layer_dropout{}".format(i),
                    nn.Dropout(p = 0.5)
                )


            self.layers.add_module(
                "mlp_layer_{}".format(i),
                nn.Linear(
                    self.model_dims[i],
                    self.model_dims[i + 1],
                ),
            )
            
            
            if(i+2 != len(self.model_dims)):
                
                    
                self.layers.add_module(
                        "mlp_act_{}".format(i),
                        self.activation)

    def reset_parameters(self):
      '''
      A method to reset the weights of the model
      '''
      for layer in self.layers:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def forward(self, x):
        return self.layers(x)
