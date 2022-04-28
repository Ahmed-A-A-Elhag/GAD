# From the DGN implementation 

import numpy as np

def scale_identity(h, D, avg_d, device):
    return h


def scale_amplification(h, D, avg_d, device):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set

    return h * ((np.log(D.cpu() + 1)).to(device) / avg_d["log"])


def scale_attenuation(h, D, avg_d, device):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set

    return h * (avg_d["log"] / (np.log(D.cpu() + 1))).to(device)
  
  
  
SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation}
