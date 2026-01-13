# src/my_project/model.py
import pytorch_lightning as pl
import torch
from torch import nn

class MyAwesomeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Model layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)
        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (preds.argmax(dim=-1) == target).float().mean()

        # log scalars
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        # log histogram to W&B if logger is available
        if self.logger:
            import wandb
            self.logger.experiment.log({'train_logits': wandb.Histogram(preds.detach().cpu())})

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (preds.argmax(dim=-1) == target).float().mean()

        # log scalars once per epoch
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
