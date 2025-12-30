import numpy as np
import torch
from torch import nn

class U_net():
    """
    U neural network
    """

    def __init__(self,
                 n_imput: int,
                 n_output: int,
                 loss_fcn: str,
                 activation_fcn: str,
                 learning_rate: float,
                 device: str = 'CUDA',
                 ):
        """
        Initialization of the U-net.

        PARAMETERS:
        ----------
        n_input : int
            Number of imput features (here it's 3, walls + entry + exit)
        n_output : int
            Number of output features (here it's 1, the predicted path)
        loss_fcn : str
            Type of loss function.
        activation_fcn : str
            Type of activation function.
        learning rate : float
            Learning rate use for the training.
        device : str
            Type of device used for the training, 'cpu' or 'CUDA' (for gpu support).
        """