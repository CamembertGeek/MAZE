import torch
from torch import nn
import torch.nn.functional as F

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
            self.get_activation_fcn(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            self.get_activation_fcn()
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
        """
        Return the appropriate loss function based on self.loss_type.
        """
        if self.loss == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Loss '{self.loss}' is not implemented.")

    def get_activation_fcn(self):
        return getattr(nn, self.activation_fcn)()

    def forward(self, x):
        """
        PARAMETER:
        ---------
        x : torch tensor
            (B, C_in, H, W) with H=100, W=101 here

        RETURN:
        ------
        (B, C_out, H, W) (Final crop to comming back to the original shape)
        """

        # The shape of the grid is 100 x 101 but we need a multiple of 16.
        # We will add a padding at the begginig and a cropat the end.

        # PAD to a multiple of 16
        B, C, H0, W0 = x.shape
        mult = 16  # = 2**4 because we have 4 maxpool (downsampling x2)

        pad_h = (mult - (H0 % mult)) % mult
        pad_w = (mult - (W0 % mult)) % mult

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h != 0 or pad_w != 0:
            # order F.pad : (left, right, top, bottom)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant", value=0.0)

        # U-NET
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


        out = self.out(x) # logits

        # CROP to come back to (H0, W0)
        if pad_h != 0 or pad_w != 0:
            out = out[..., pad_top:pad_top + H0, pad_left:pad_left + W0]

        return out

    def get_optimizer(self):
        """
        Return optimizer instance.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def compute_metrics(self, logits, targets, threshold=0.5, eps=1e-7):
        """
        Compute simple segmentation metrics (Dice + IoU) on a batch.

        We apply sigmoid on logits -> probabilities -> threshold -> binary mask.

        RETURNS:
        --------
        dict with 'dice' and 'iou'
        """
        if targets.dtype != torch.float32:
            targets = targets.float()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # Flatten per sample (and per channel)
        # shape: (B, C, H*W)
        preds_f = preds.flatten(start_dim=2)
        targets_f = targets.flatten(start_dim=2)

        intersection = (preds_f * targets_f).sum(dim=2)          # (B, C)
        pred_sum = preds_f.sum(dim=2)                             # (B, C)
        target_sum = targets_f.sum(dim=2)                         # (B, C)
        union = pred_sum + target_sum - intersection              # (B, C)

        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        iou = (intersection + eps) / (union + eps)

        # Average on batch + canaux
        return {
            "dice": float(dice.mean().item()),
            "iou": float(iou.mean().item()),
        }

    def train_model(self, train_loader, val_loader=None, epochs=10, threshold=0.5, grad_clip=0.1, verbose=True):
        """
        Train loop.

        PARAMETERS:
        ----------
        train_loader : DataLoader
        val_loader : DataLoader or None
        epochs : int
        threshold : float (for metrics)
        grad_clip : float or None
        verbose : bool

        RETURNS:
        --------
        history : list of dicts (train/val stats)
        """

        history = []

        for epoch in range(1 + epochs + 1):
            self.train()

            total_loss = 0.0
            n_batch = 0.0
            dice_sum = 0.0
            iou_sum = 0.0

            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self(x)
                loss = self.compute_loss(logits=logits, targets=y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                metrics = self.compute_metrics(logits.detach(), y, threshold=threshold)

                total_loss += float(loss.item())
                dice_sum += metrics["dice"]
                iou_sum += metrics["iou"]
                n_batches += 1

                train_stats = {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "train_dice": dice_sum / max(n_batches, 1),
                "train_iou": iou_sum / max(n_batches, 1),
            }

            if val_loader is not None:
                val_stats = self.validate_epoch(val_loader, threshold=threshold)
                train_stats.update({
                    "val_loss": val_stats["loss"],
                    "val_dice": val_stats["dice"],
                    "val_iou": val_stats["iou"],
                })

            history.append(train_stats)

            if verbose:
                if val_loader is None:
                    print(f"Epoch {epoch:03d} | loss={train_stats['train_loss']:.4f} | dice={train_stats['train_dice']:.4f} | iou={train_stats['train_iou']:.4f}")
                else:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train loss={train_stats['train_loss']:.4f}, dice={train_stats['train_dice']:.4f}, iou={train_stats['train_iou']:.4f} | "
                        f"val loss={train_stats['val_loss']:.4f}, dice={train_stats['val_dice']:.4f}, iou={train_stats['val_iou']:.4f}"
                    )

        return history

    def compute_loss(self, logits, targets):
        """
        Compute the loss from logits and targets.

        PARAMETERS:
        ----------
        logits : torch.Tensor
            (B, C_out, H, W) - output of the network (logits)
        targets : torch.Tensor
            (B, C_out, H, W) - ground truth mask (0/1) float
        
        RETURN:
        ------
        torch.Tensor : scalar loss
        """

        # Security : BCEWithLogitsLoss is waiting for a float.
        if targets.dtype != torch.float32:
            targets = targets.float()

        return self.loss_fcn(logits, targets)

    def validate_epoch(self, loader):
        pass

    @torch.no_grad()
    def predict_proba(self, x):
        """
        Predict probabilities (sigmoid(logits)).

        PARAMETER:
        ---------
        x : torch.Tensor (B, C_in, H, W)

        RETURN:
        ------ 
        torch.Tensor (B, C_out, H, W) in [0,1]
        """
        self.eval()

        x = x.to(self.device, non_blocking=True)
        logits = self(x)

        return torch.sigmoid(logits)
        
    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        """
        Predict binary mask with a threshold.

        RETURN:
        ------ 
        torch.Tensor float {0,1} (B, C_out, H, W)
        """
        probs = self.predict_proba(x)

        return (probs >= threshold).float()
        
    @torch.no_grad()
    def evaluate(self, loader, threshold=0.5):
        """
        Alias of validate_epoch (same metrics).
        """

        return self.validate_epoch(loader, threshold=threshold)

    def save_model(self, path: str = "model.pth"):
        """
        Save the model weights to a file.
    
        PARAMETER:
        ----------
        path : str
            Path to the output file (default: 'model.pth')
        """
        torch.save(self.state_dict(), path)
        
    def load_model(self, path: str):
        """
        Load model weights from a file.
    
        PARAMETER:
        ----------
        path : str
            Path to the file where weights were saved
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
