import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


class SubsetSC(SPEECHCOMMANDS):
    """Custom subset of Speech Commands dataset with only 'yes' and 'no' classes"""
    
    def __init__(self, subset: str = None, root: str = "./", download: bool = True):
        super().__init__(root, download=download)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                res = []
                for line in fileobj:
                    if 'yes/' in line or 'no/' in line:
                        res.append(os.path.normpath(os.path.join(self._path, line.strip())))
            return res

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            # Filter only yes/no for training
            self._walker = [rec for rec in self._walker if ('yes/' in rec or 'no/' in rec)]
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        elif subset is None:
            # Keep all files for full dataset
            pass


class SpeechCommandsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Speech Commands dataset.
    Handles data loading, preprocessing, and splitting.
    """
    
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        download: bool = True,
        train_subset: str = "training",
        val_subset: str = "validation",
        test_subset: str = "testing",
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset
        
        # Will be set during setup
        self.labels = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
    def prepare_data(self):
        """
        Download data if needed.
        Called only once and on only one GPU.
        """
        # Initialize dataset to trigger download if needed
        SubsetSC(root=self.data_dir, download=self.download)
    
    def setup(self, stage=None):
        """
        Setup datasets for each stage (fit, test, predict).
        Called on every GPU.
        """
        # Create datasets for each split
        if stage == "fit" or stage is None:
            self.train_set = SubsetSC(
                subset=self.train_subset,
                root=self.data_dir,
                download=self.download
            )
            
            # Create validation set
            self.val_set = SubsetSC(
                subset=self.val_subset,
                root=self.data_dir,
                download=self.download
            )
            
            # Extract labels from training set
            self.labels = sorted(['yes', 'no']) # sorted(list(set(datapoint[2] for datapoint in self.train_set)))
            print(f"Found {len(self.labels)} classes: {self.labels}")
        
        if stage == "test" or stage is None:
            self.test_set = SubsetSC(
                subset=self.test_subset,
                root=self.data_dir,
                download=self.download
            )
            
            # If labels not already set, extract from test set
            if self.labels is None:
                self.labels = sorted(['yes', 'no']) #sorted(list(set(datapoint[2] for datapoint in self.test_set)))
    
    def label_to_index(self, word: str) -> torch.Tensor:
        """Convert label word to index tensor"""
        return torch.tensor(self.labels.index(word))
    
    def index_to_label(self, index: int) -> str:
        """Convert index to label word"""
        return self.labels[index]
    
    @staticmethod
    def pad_sequence(batch):
        """
        Pad a list of variable length tensors to the same length.
        
        Args:
            batch: List of tensors of shape [channels, time]
        
        Returns:
            Padded tensor of shape [batch, channels, max_time]
        """
        # Transpose to [time, channels] for pad_sequence
        batch = [item.t() for item in batch]
        # Pad sequences
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        # Transpose back to [batch, channels, time]
        return batch.permute(0, 2, 1)
    
    def collate_fn(self, batch):
        """
        Custom collate function for Speech Commands dataset.
        
        Args:
            batch: List of tuples (waveform, sample_rate, label, speaker_id, utterance_number)
        
        Returns:
            tensors: Padded waveforms of shape [batch, channels, max_time]
            targets: Label indices of shape [batch]
        """
        tensors, targets = [], []
        
        for waveform, _, label, *_ in batch:
            tensors.append(waveform)
            targets.append(self.label_to_index(label))
        
        # Pad waveforms to same length
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)
        
        return tensors, targets
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# Example usage with automatic device detection and worker configuration
if __name__ == "__main__":
    # Auto-detect device and configure workers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        num_workers = 4  # You can adjust this based on your CPU cores
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    
    # Initialize data module
    dm = SpeechCommandsDataModule(
        data_dir="./data",
        batch_size=256,
        num_workers=num_workers,
        pin_memory=pin_memory,
        download=True,
    )
    
    # Setup for training
    dm.setup(stage="fit")
    
    # Access labels
    print(f"Classes: {dm.labels}")
    
    # Test dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Example batch
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Targets: {targets}")
        
        # Convert indices back to labels
        labels = [dm.index_to_label(idx.item()) for idx in targets[:5]]
        print(f"  First 5 labels: {labels}")
        break