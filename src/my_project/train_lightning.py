# src/my_project/train.py
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import MyAwesomeModel
from data import corrupt_mnist  # your dataset loader

def main():
    # 1️⃣ Load dataset
    train_set, val_set = corrupt_mnist()
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    # 2️⃣ Instantiate model
    model = MyAwesomeModel()

    # 3️⃣ Callbacks
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best_model",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    # 4️⃣ Logger (W&B)
    wandb_logger = WandbLogger(project="dtu_mlops")

    # 5️⃣ Trainer
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=0.2,  # use 20% of training data
        limit_val_batches=0.2,    # use 20% of validation data
        accelerator="auto",
        devices=1,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=wandb_logger,
        default_root_dir="lightning_logs"
    )

    # 6️⃣ Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
