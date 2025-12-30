import numpy as np
import torch
from torch import nn

"""
Note:
The UNet model was introduced in the paper “U-Net: Convolutional Networks for Biomedical
Image Segmentation” by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
"""

class Unet(nn.Module):
    """
    U neural network
    """

    def __init__(self,
                 n_input: int,
                 n_output: int,
                 loss: str,
                 activation_fcn: str,
                 learning_rate: float,
                 device: str = 'cuda',
                 ):
        """
        Initialization of the U-net.

        PARAMETERS:
        ----------
        n_input : int
            Number of imput features (here it's 3, walls + entry + exit)
        n_output : int
            Number of output features (here it's 1, the predicted path)
        loss : str
            Type of loss function.
        activation_fcn : str
            Type of activation function.
        learning rate : float
            Learning rate use for the training.
        device : str
            Type of device used for the training, 'cpu' or 'CUDA' (for gpu support).

        Architecture of the U-net:
        --------------------------
        INPUT : Tensor (B, C_in, H, W), (here C_in = 3 : walls / entry / exit)
        OUTPUT: Tensor (B, C_out, H, W), (here C_out = 1 : predicted path mask)

        Core building block (DoubleConv):
        --------------------------------
        1) Conv 3x3 -> Normalisation (BatchNorm / GroupNorm) -> Activation
        2) Conv 3x3 -> Normalisation -> Activation

        Encoder (contracting path):
        ---------------------------
        At each level:
            1) DoubleConv
            2) Downsampling x2 (MaxPool 2x2 or strided conv)
            3) Channels are doubled

        Typical channel progression (example with base = 32):
            Level 0 (H,   W)   : C_in   -> 32   (save skip_0)
            Level 1 (H/2, W/2) : 32     -> 64   (save skip_1)
            Level 2 (H/4, W/4) : 64     -> 128  (save skip_2)
            Level 3 (H/8, W/8) : 128    -> 256  (save skip_3)

        Bottleneck:
        -----------
            DoubleConv with the maximum number of channels
            Example: 256 -> 512  (resolution H/16, W/16 if you did 4 downs)
        
            Decoder (expansive path):
        -------------------------
        At each level:
            1) Upsampling x2 (ConvTranspose2d or Upsample + Conv)
            2) Concatenate with the corresponding skip connection (same resolution)
            3) DoubleConv
            4) Channels are divided by 2

        Typical decoder steps (mirror of the encoder):
            Up from bottleneck to Level 3 resolution:
                upsample -> concat(skip_3) -> DoubleConv  (512 -> 256)
            Up to Level 2 resolution:
                upsample -> concat(skip_2) -> DoubleConv  (256 -> 128)
            Up to Level 1 resolution:
                upsample -> concat(skip_1) -> DoubleConv  (128 -> 64)
            Up to Level 0 resolution:
                upsample -> concat(skip_0) -> DoubleConv  (64 -> 32)

        Output head:
        ------------
            Final Conv 1x1 : maps last feature channels -> C_out
            Example: 32 -> 1

        Notes:
        ------
        - Skip connections are essential to recover fine spatial details (thin paths in mazes).
        - If H or W are not divisible by 2^depth, padding/cropping may be needed during concatenation.
        - For binary segmentation, the network often outputs logits; sigmoid is applied for probabilities/visualization.
        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.loss = loss
        self.activation_fcn = activation_fcn
        self.learning_rate = learning_rate
        self.device = device

        self._build_model()

        self.to(self.device)

        self.loss_fcn = self.get_loss_fcn()

        self.optimizer = self.get_optimizer()




    def double_conv(self, in_channel, out_channel):
        conv_op = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv_op

    def _build_model(self):

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = self.double_conv(self.n_input, 64)
        self.down_convolution_2 = self.double_conv(64, 128)
        self.down_convolution_3 = self.double_conv(128, 256)
        self.down_convolution_4 = self.double_conv(256, 512)
        self.down_convolution_5 = self.double_conv(512, 1024)

        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2, 
            stride=2)
        
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = self.double_conv(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2, 
            stride=2)
        self.up_convolution_2 = self.double_conv(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2, 
            stride=2)
        self.up_convolution_3 = self.double_conv(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2, 
            stride=2)
        self.up_convolution_4 = self.double_conv(128, 64)

        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=self.n_output, 
            kernel_size=1
        ) 

        

    def get_loss_fcn(self):
        pass

    def get_activation_fcn(self):
        pass

    def forward(self, x):

        down_1 = self.down_convolution_1(x)

        down_2 = self.max_pool2d(down_1)

        down_3 = self.down_convolution_2(down_2)

        down_4 = self.max_pool2d(down_3)

        down_5 = self.down_convolution_3(down_4)

        down_6 = self.max_pool2d(down_5)

        down_7 = self.down_convolution_4(down_6)

        down_8 = self.max_pool2d(down_7)

        down_9 = self.down_convolution_5(down_8)        
        # *** DO NOT APPLY MAX POOL TO down_9 ***
        

        up_1 = self.up_transpose_1(down_9)

        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))

        up_2 = self.up_transpose_2(x)

        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))

        up_3 = self.up_transpose_3(x)

        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

        up_4 = self.up_transpose_4(x)

        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))


        out = self.out(x)

        return out

    def get_optimizer(self):
        pass

    def compute_metrics(self):
        pass

    def train_model(self):
        pass

    def compute_loss(self):
        pass

    def validate_epoch(self, loader):
        pass

    def predict_proba(self):
        pass
        
    def predict(self):
        pass
        
    def evaluate(self):
        pass

    def save_model(self, path: str = "model.pth"):
        """
        Save the model weights to a file.
    
        PARAMETERS:
        -----------
        path : str
            Path to the output file (default: 'model.pth')
        """
        torch.save(self.state_dict(), path)
        
    def load_model(self, path: str):
        """
        Load model weights from a file.
    
        PARAMETERS:
        -----------
        path : str
            Path to the file where weights were saved
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
