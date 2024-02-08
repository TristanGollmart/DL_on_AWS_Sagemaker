import json
from pathlib import Path
from typing import Tuple, Callable, List, Any

import pandas as pd
import torch
import torch.optim
from PIL import Image
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wandb import log
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

ROOTPATH = Path("../data/cassava")
IMAGE_SIZE = (128, 128)
MEAN = [0.4766, 0.4527, 0.2936]
STD = [0.2275, 0.2224, 0.2210]
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 3
device = 'gpu' if torch.cuda.is_available() else 'cpu'


class Data(Dataset):
    def __init__(self, df: pd.DataFrame, transforms = None):
        self.X = [ROOTPATH / "train_images" / filename for filename in df["image_id"].values]
        self.y = df['label'].values.tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = Image.open(self.X[idx])
        y = self.y[idx]
        if self.transforms is not None:
            X = self.transforms(X)
        return X, y

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.linear1 = nn.Linear(self.base.fc.in_features, self.base.fc.in_features // 2)
        self.linear2 = nn.Linear(self.base.fc.in_features // 2, num_classes)
        self.norm1 = nn.BatchNorm1d(self.base.fc.in_features)
        self.norm2 = nn.BatchNorm1d(self.base.fc.in_features // 2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.base.fc = nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.dropout1(self.norm1(F.leaky_relu(self.base(x))))
        out = self.dropout2(self.norm2(F.leaky_relu(self.linear1(out))))
        out = self.linear2(out)
        return out

class LightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, loss_fn: Callable):
        super().__init__() # enherit methods
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn

    def common_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        return loss, acc

    def training_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args: List[Any]):
        loss, acc = self.common_step(batch)
        self.log(name="training loss", value=loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(name="training accuracy", value=acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def on_train_epoch_end(self, *args):
        if self.current_epoch == 0:
            for p in self.model.parameters():
                p.requires_grad = True

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args: List[Any]):
        loss, acc = self.common_step(batch)
        self.log(name="validation loss", value=loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(name="validation accuracy", value=acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

# data processing
df = pd.read_csv(ROOTPATH / "train.csv")
with open(ROOTPATH / "label_num_to_disease_map.json") as f:
    label_map = json.load(f)
label_map = {int(i): v for i, v in label_map.items()}
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'].values)
print(df["label"].map(label_map).value_counts())

train_transforms = Compose([Resize(IMAGE_SIZE),
                      RandomHorizontalFlip(),
                      RandomRotation(10),
                      ToTensor(),
                      Normalize(MEAN, STD)])


test_transforms = Compose([Resize(IMAGE_SIZE),
                      ToTensor(),
                      Normalize(MEAN, STD)])

train_ds = Data(train_df, train_transforms)
val_ds = Data(val_df, test_transforms)

train_dl = DataLoader(train_ds,
                      BATCH_SIZE,
                      drop_last=True,
                      shuffle=True)

val_dl = DataLoader(val_ds,
                    batch_size=64,
                    drop_last=False,
                    shuffle=False)

# pl trainer

model = Model(num_classes=df['label'].nunique())
label_counts = df["label"].value_counts().sort_index()
class_weights = label_counts.max()/label_counts
loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
lightning_model = LightningModule(model, LR, loss_fn)

# wandb logger
logger = WandbLogger(project="Cassava-Leaves", save_dir="../logs/cassava/", name='cassava-1')
logger.experiment.config.update({"lr": LR, "epochs": EPOCHS, "batchsize": BATCH_SIZE})
trainer = pl.Trainer(max_epochs=EPOCHS,
                     logger=logger,
                     gradient_clip_val=1.0,
                     precision=16)

trainer.fit(lightning_model, train_dl, val_dl)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
    'lr_sched': scheduler.state_dict()
}
torch.save(checkpoint, ROOTPATH / "models" / "cassava" / f"resnet34_epoch{EPOCHS}")
print("finished")
