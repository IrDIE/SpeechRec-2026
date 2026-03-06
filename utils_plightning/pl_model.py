import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils_plightning.pl_utils_data import SpeechCommandsDataModule
from clearml import Task

class M5(pl.LightningModule):

    def __init__(
        self,
        n_input=1,
        n_output=35,
        stride=1,
        n_groups=1,
        n_channel=32,
        learning_rate=1e-3,
        transform=None,
        experiment_name="default_exp"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])

        self.n_input = n_input
        self.n_output = n_output
        self.stride = stride
        self.n_channel = n_channel
        self.learning_rate = learning_rate
        self.transform = transform

        self.n_groups = n_groups  # Store the groups parameter
        self.experiment_name = experiment_name
        
        # Model architecture with configurable groups
        # Note: n_groups must divide n_channel for conv1, conv2 and 2*n_channel for conv3, conv4
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(2)
        
        # Grouped convolutions for subsequent layers
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_train_start(self):
        task = Task.current_task()
        if task:
            for k, v in self.hparams.items():
                if isinstance(v, (int, float)):
                    task.logger.report_scalar(
                        title="hyperparameters",
                        series=k,
                        value=v,
                        iteration=0   # single step
                    )
                
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

    def training_step(self, batch, batch_idx):
        data, target = batch
        
        if self.transform:
            data = self.transform(data)
        
        output = self(data)
        loss = F.nll_loss(output.squeeze(), target)
        
        # Calculate accuracy
        pred = output.argmax(dim=2).squeeze()
        correct = (pred == target).float().mean()
        accuracy = correct * 100
        
        # Log metrics with experiment name as prefix for better organization
        self.log(f'loss/train_loss', loss, 
                on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'accuracy/train_accuracy', accuracy, 
                on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        
        if self.transform:
            data = self.transform(data)
        
        output = self(data)
        loss = F.nll_loss(output.squeeze(), target)
        
        pred = output.argmax(dim=2).squeeze()
        correct = (pred == target).float().mean()
        accuracy = correct * 100
        
        # Log validation metrics with experiment name prefix
        self.log(f'loss/val_loss', loss, 
                on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'accuracy/val_accuracy', accuracy, 
                on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        data, target = batch
        
        if self.transform:
            data = self.transform(data)
        
        output = self(data)
        loss = F.nll_loss(output.squeeze(), target)
        
        pred = output.argmax(dim=2).squeeze()
        correct = (pred == target).float().mean()
        accuracy = correct * 100
        
        # Log test metrics with experiment name prefix
        self.log(f'loss/test_loss', loss, 
                on_step=False, on_epoch=True, logger=True)
        self.log(f'accuracy/test_accuracy', accuracy, 
                on_step=False, on_epoch=True, logger=True)
        
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        """Log additional epoch-level information"""
        # You can add custom epoch-end logic here
        pass


# Custom callback for logging hyperparameters
class HyperparameterLogger(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        """Log hyperparameters at the start of training"""
        hparams = pl_module.hparams
        trainer.logger.log_hyperparams(hparams)