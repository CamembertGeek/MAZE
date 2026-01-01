import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DatasetBuilder():

    def __init__(self,
                 training_file_path: str,
                 solved_file_path: str,
                 split: float = 0.8,
                 seed: int = 42,
                 device: str = 'cuda',
                 ):
        """
        Initialization of the Dataset Builder.

        PARAMETERS:
        ----------
        training_file_path : str
            Path trought the training file.
        solved_file_path : str
            Path trought the solution of the training file.
        split : float
            The proportion of the split between the training and validation set.
        seed : int
            Seed of the slplit, for reprodictibility.
        device : str
            Type of device used for the training, 'cpu' or 'cuda' (for gpu support).    
        """
        
        self.training_file_path = training_file_path
        self.solved_file_path = solved_file_path
        self.split = split
        self.seed = seed
        self.device = device

    def load_data(self):
        """
        Load data from the directory.

        RETURNS:
        -------
        grids : dict
            Dictionary containing the trainig grids, the entry location and the exit location.
        solved_grids : dict
            Dictionary containing the solution of the training grids.
        """
        grids = torch.load(self.training_file_path, map_location="cpu")

        solved_grids = torch.load(self.solved_file_path, map_location="cpu")

        return grids, solved_grids

    def make_split(self, grids_dict: dict):
        """
        Make the slipt between training and validation set.

        PARAMETER:
        ---------
        grids_dict : dict
            Dictionary of all the training grid, only use to have the IDs of the grids.

        RETURN:
        ------
        train_IDs : list[str]
            List of the IDs use for the training.
        val_IDs : list[str]
            List of the IDs use for the validation.
        """
        IDs = sorted(list(grids_dict.keys()))

        if self.seed is not None:
            random.seed(self.seed)

        random.shuffle(IDs)

        len_train_ids = int(self.split * len(IDs))

        train_IDs = list(IDs[:len_train_ids])
        val_IDs = list(IDs[len_train_ids:])

        return train_IDs, val_IDs


    def build_datasets(self):
        """
        Build the training and validation dataset.
        """
        grids, solved = self.load_data()
        train_ids, val_ids = self.make_split(grids)

        self.train_dataset = MazeDataset(grids, solved, train_ids)
        self.val_dataset = MazeDataset(grids, solved, val_ids)

    def get_dataloaders(self, batch_size, num_workers=0, pin_memory=None):
        """
        Docstring for get_dataloaders
        
        PARAMETERS:
        ----------
        batch_size : int
            Size of the batch.
        num_workers : int
            Number of workers
        pin_memory :

        RETURNS:
        -------
        train_loader : 

        val_loader : 

        """
        self.build_datasets()

        if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
            raise AttributeError("You must build_datasets() before calling get_dataloaders().")

        
        if pin_memory is None:
            pin_memory = (self.device == "cuda")

        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

        return train_loader, val_loader

    def sanity_check(self, n=1, batch_size=4):
        """
        Run quick checks to ensure datasets/dataloaders are coherent.

        Checks:
        - train/val datasets exist
        - sample shapes/dtypes/values
        - entry/exit masks are one-hot
        - solution mask is binary and non-empty
        - (optional) one DataLoader batch stacks correctly
        """
        # Build datasets if not already done
        if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
            self.build_datasets()

        print("=== SANITY CHECK ===")
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size  : {len(self.val_dataset)}")

        # Check a few individual samples
        n = int(n)
        n = max(1, n)
        n = min(n, len(self.train_dataset))

        for i in range(n):
            x, y = self.train_dataset[i]

            # Basic type/shape checks
            assert isinstance(x, torch.Tensor), f"x is not a Tensor at i={i}"
            assert isinstance(y, torch.Tensor), f"y is not a Tensor at i={i}"

            assert x.ndim == 3 and x.shape[0] == 3, f"Bad x shape at i={i}: {tuple(x.shape)} (expected (3,H,W))"
            assert y.ndim == 3 and y.shape[0] == 1, f"Bad y shape at i={i}: {tuple(y.shape)} (expected (1,H,W))"

            H, W = x.shape[1], x.shape[2]
            assert y.shape[1] == H and y.shape[2] == W, f"x/y spatial mismatch at i={i}: x={(H,W)}, y={(y.shape[1], y.shape[2])}"

            assert x.dtype == torch.float32, f"x dtype must be float32 (BCE expects float). Got {x.dtype} at i={i}"
            assert y.dtype == torch.float32, f"y dtype must be float32 (BCE expects float). Got {y.dtype} at i={i}"

            # Value checks (binary-ish)
            x0_min, x0_max = float(x[0].min().item()), float(x[0].max().item())
            y_min, y_max = float(y.min().item()), float(y.max().item())
            assert x0_min >= 0.0 and x0_max <= 1.0, f"walls channel not in [0,1] at i={i}: min={x0_min}, max={x0_max}"
            assert y_min >= 0.0 and y_max <= 1.0, f"solution not in [0,1] at i={i}: min={y_min}, max={y_max}"

            # One-hot checks for entry/exit
            entry_sum = float(x[1].sum().item())
            exit_sum = float(x[2].sum().item())
            assert abs(entry_sum - 1.0) < 1e-6, f"entry_mask sum != 1 at i={i} (sum={entry_sum})"
            assert abs(exit_sum - 1.0) < 1e-6, f"exit_mask sum != 1 at i={i} (sum={exit_sum})"

            # Non-empty path check (can be 0 if something went wrong)
            path_sum = float(y.sum().item())
            assert path_sum > 0.0, f"solution seems empty at i={i} (sum={path_sum})"

        print(f"✓ {n} sample(s) look OK.")

        # Check one batch stacks correctly (optional but useful)
        train_loader, _ = self.get_dataloaders(batch_size=batch_size, num_workers=0)
        xb, yb = next(iter(train_loader))

        assert xb.ndim == 4 and xb.shape[1] == 3, f"Bad batch x shape: {tuple(xb.shape)} (expected (B,3,H,W))"
        assert yb.ndim == 4 and yb.shape[1] == 1, f"Bad batch y shape: {tuple(yb.shape)} (expected (B,1,H,W))"
        assert xb.dtype == torch.float32 and yb.dtype == torch.float32, f"Batch dtype must be float32. Got {xb.dtype}, {yb.dtype}"

        print(f"✓ DataLoader batch OK: x={tuple(xb.shape)} y={tuple(yb.shape)}")
        print("=== SANITY CHECK PASSED ===")


    def summary(self, show_example=True):
        """
        Print a short summary of the dataset builder and datasets.
        """
        if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
            self.build_datasets()

        print("=== DATASET SUMMARY ===")
        print(f"Training file : {self.training_file_path}")
        print(f"Solved file   : {self.solved_file_path}")
        print(f"Split         : {self.split} (seed={self.seed})")
        print(f"Device        : {self.device}")
        print(f"Train size    : {len(self.train_dataset)}")
        print(f"Val size      : {len(self.val_dataset)}")

        if show_example and len(self.train_dataset) > 0:
            x, y = self.train_dataset[0]
            print("--- Example (train_dataset[0]) ---")
            print(f"x shape/dtype : {tuple(x.shape)} / {x.dtype}")
            print(f"y shape/dtype : {tuple(y.shape)} / {y.dtype}")
            print(f"walls min/max : {float(x[0].min().item())} / {float(x[0].max().item())}")
            print(f"entry sum     : {float(x[1].sum().item())}")
            print(f"exit sum      : {float(x[2].sum().item())}")
            print(f"y min/max     : {float(y.min().item())} / {float(y.max().item())}")
            print(f"path sum      : {float(y.sum().item())}")

        print("=== END SUMMARY ===")




class MazeDataset(Dataset):
    """
    PyTorch Dataset for maze segmentation.

    Each sample is defined by an ID:
      - grids_dict[ID] = {'grid': (H,W) uint8 0/1, 'entry': (2,), 'exit': (2,)}
      - solved_dict[ID] = {'solution': (H,W) uint8 0/1}

    Returns:
      x : (3, H, W) float32  -> [walls, entry_mask, exit_mask]
      y : (1, H, W) float32  -> [solution]
    """

    def __init__(self, grids_dict: dict, solved_dict: dict, ids: list[str], return_id: bool = False):
        self.grids = grids_dict
        self.solved = solved_dict
        self.ids = list(ids)
        self.return_id = return_id

        # Optional sanity check: keys match
        # (comment out if you want faster init)
        missing = [k for k in self.ids if k not in self.solved]
        if missing:
            raise KeyError(f"{len(missing)} ids are missing in solved_dict (example: {missing[0]}).")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        maze_id = self.ids[idx]

        sample = self.grids[maze_id]
        target = self.solved[maze_id]

        grid = sample["grid"]          # (H,W) uint8
        entry = sample["entry"]        # (2,)  [row, col]
        exit_ = sample["exit"]         # (2,)  [row, col]
        solution = target["solution"]  # (H,W) uint8

        # Ensure tensors (in case entry/exit are not)
        if not torch.is_tensor(grid):
            grid = torch.tensor(grid)
        if not torch.is_tensor(entry):
            entry = torch.tensor(entry)
        if not torch.is_tensor(exit_):
            exit_ = torch.tensor(exit_)
        if not torch.is_tensor(solution):
            solution = torch.tensor(solution)

        H, W = grid.shape[-2], grid.shape[-1]

        # Build entry/exit masks
        entry_mask = torch.zeros((H, W), dtype=torch.float32)
        exit_mask = torch.zeros((H, W), dtype=torch.float32)

        er, ec = int(entry[0].item()), int(entry[1].item())
        xr, xc = int(exit_[0].item()), int(exit_[1].item())

        # Security clamp (optional)
        if not (0 <= er < H and 0 <= ec < W):
            raise ValueError(f"Entry coords out of bounds for id={maze_id}: entry={entry.tolist()}, grid_shape={(H,W)}")
        if not (0 <= xr < H and 0 <= xc < W):
            raise ValueError(f"Exit coords out of bounds for id={maze_id}: exit={exit_.tolist()}, grid_shape={(H,W)}")

        entry_mask[er, ec] = 1.0
        exit_mask[xr, xc] = 1.0

        # x: (3,H,W)
        walls = grid.to(torch.float32)
        x = torch.stack([walls, entry_mask, exit_mask], dim=0)

        # y: (1,H,W)
        y = solution.to(torch.float32).unsqueeze(0)

        if self.return_id:
            return x, y, maze_id
        return x, y