from typing import Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mlimg

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 2
USE_LIGHTNING = True

class Data(Dataset):
    def __init__(self, df: pd.DataFrame, transforms):
        self.files = df['filepaths'].values
        self.y = df['y'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.transforms(torchvision.io.read_image(self.files[idx]))/255.0, self.y[idx]


class Model(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super().__init__()
        self.base = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        self.bn1 = nn.BatchNorm1d(self.base.fc.in_features)
        self.linear = nn.Linear(self.base.fc.in_features, num_classes)
        self.base.fc = nn.Identity()

        if freeze:
            self.base = self.base.eval()
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.linear(self.bn1(self.base(x)))

class LightningModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, loss_fn: Callable):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.accuracy = lambda x, y: (x.argmax(axis=-1) == y).float().mean()

    def common_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor]):
        x, y = batch
        y = y.long()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, *args):
        loss, acc = self.common_step(batch)
        self.log("training_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("training_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, *args):
        loss, acc = self.common_step(batch)
        self.log("validation_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("validation_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), self.lr)

# data processing
df = pd.read_csv("../data/birds/birds.csv")
df.head()

le = LabelEncoder()
df['y'] = le.fit_transform(df['labels'])
path_prefix = '../data/birds/'
df['filepaths'] = df['filepaths'].str.replace("AKULET", "AUKLET")
df['filepaths'] = df['filepaths'].str.replace("  AUKLET", " AUKLET")
df['filepaths'] = path_prefix + df['filepaths']
print(df["labels"].value_counts())

transforms = Compose([Resize((224, 224))])
train_dl = DataLoader(Data(df[df['data set']=='train'], transforms=transforms), shuffle=True, batch_size=BATCH_SIZE)
val_dl = DataLoader(Data(df[df['data set']=='valid'], transforms=transforms), shuffle=False, batch_size=BATCH_SIZE)
test_dl = DataLoader(Data(df[df['data set']=='test'], transforms=transforms), shuffle=False, batch_size=BATCH_SIZE)

# test data loading
for i, (x, y) in enumerate(val_dl):
    print(i, y)


# standard pytorch training:
if not USE_LIGHTNING:
    device = "gpu" if torch.cuda.is_available() else "cpu"
    num_classes = len(le.classes_)
    model = Model(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(train_dl):
            x = x.to(device)
            y = y.long().to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i%100==0:
                print(f"step {i}, loss {loss.item():.3f}")
else:
    # training
    device = "gpu" if torch.cuda.is_available() else "cpu"
    num_classes = len(le.classes_)
    trainer = pl.Trainer(max_epochs=EPOCHS)
    model = Model(num_classes)
    lightning_model = LightningModel(model, lr=LR, loss_fn=nn.CrossEntropyLoss())
    trainer.fit(model=lightning_model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# testset eval:
model.eval().to(device)

acc = []
for x, y in test_dl:
    y.to(device)
    y_pred = model(x.to(device)).to(device)
    acc.append((y_pred.argmax(axis=-1)==y).float().tolist())
print("test set accuracy: {:.3f}".format(np.mean(np.concatenate(acc))))

print('finished')