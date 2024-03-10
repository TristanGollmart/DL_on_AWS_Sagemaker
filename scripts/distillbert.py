import torch.optim
from transformers import AutoTokenizer, AutoModel
import pathlib
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor

ROOT_PATH = pathlib.Path("..")
DATA_PATH = ROOT_PATH / "data"
MODEL_PATH = ROOT_PATH / "models"
device = 'cpu'
MODEL = 'distilbert-base-multilingual-cased'
BATCH_SIZE = 32
EPOCHS = 1
MAX_DOC_LENGTH = 256
LR = 5e-5
RELOAD = False


tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoModel.from_pretrained(MODEL).config # information about model configuration -> use to attach correct classifier head

train_df = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "train.csv")[['comment_text', 'toxic']]
test_df = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "test.csv")
test_labels = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "test_labels.csv")

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['toxic'].values)
print(train_df['toxic'].value_counts())
print(val_df['toxic'].value_counts())
print(train_df.sample(5))

pos_weight = Tensor([train_df['toxic'].value_counts()[0]/train_df['toxic'].value_counts()[1]])

# analyze comments labeled as -1 in test set
label_ids = test_labels['id'][test_labels['toxic'] == -1]
print(test_df[test_df['id'].isin(label_ids)]['comment_text'][:10])
# ==> It seems like the -1 comments are not so clear and sometime have some special symbols included, drop them

class Train_Dataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.X = df['comment_text'].values
        self.y = df['toxic'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Test_Dataset(Dataset):
    def __init__(self, df_comments, df_labels):
        super().__init__()
        self.X_df = df_comments['comment_text']
        self.X_df.index = df_comments['id']
        ids = [df_labels['id'].iloc[i] for i in range(len(df_labels['id'])) if df_labels['toxic'].iloc[i]>=0]
        self.X = [self.X_df[id] for id in ids]
        # drop -1 values
        self.y_df = df_labels['toxic']
        self.y_df.index = df_labels['id']
        self.y = [self.y_df[id] for id in ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dl = DataLoader(Train_Dataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(Train_Dataset(val_df), batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(Test_Dataset(test_df, test_labels), batch_size=1, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = AutoModel.from_pretrained(MODEL)
        self.head = nn.Linear(self.base.config.dim, 1)

    def forward(self, input_ids, attention_mask):
        sequence_out = self.base(input_ids, attention_mask).last_hidden_state[:, 0, :] #The CLS token to classify is always the first of the sequence
        return self.head(sequence_out)

def evaluate(model, dataloader, loss_fn, tokenizer, device):
    with torch.no_grad():
        model.eval()
        losses = []
        accuracies = []
        preds = []
        y_true= []

        for (i, batch) in enumerate(dataloader):
            x, y = batch
            y.to(device)
            encodings = tokenizer(x, max_length=MAX_DOC_LENGTH, return_tensors='pt', padding=True, truncation=True)
            logits = model(**encodings).to(device)
            losses.append(loss_fn(logits.squeeze(), y.float()).item())

            probs = torch.sigmoid(logits)
            ypred = (probs > THRESH)
            acc = (ypred == y).float().mean()
            accuracies.append(acc)

            preds += ypred.int().squeeze().tolist()
            y_true += y.tolist()

            if i>10:
                break

    cf = confusion_matrix(y_true, preds)
    return np.mean(losses), np.mean(accuracies), cf

model = Model()

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(params=model.parameters(), lr=LR)
THRESH = 0.5

if not RELOAD:
    all_losses = []
    all_accuracies = []
    for epochs in range(EPOCHS):
        model.to(device)
        model.train()
        for (i, batch) in enumerate(train_dl):
            x, y = batch
            #x.to(device)
            y.to(device)
            encodings = tokenizer(x, max_length = MAX_DOC_LENGTH, return_tensors = 'pt', padding=True, truncation=True)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            logits = model(input_ids, attention_mask).to(device)
            loss = loss_fn(logits.squeeze(), y.float())
            loss.backward()
            opt.step()
            opt.zero_grad()
            all_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            ypred = (probs > THRESH)
            acc = (ypred == y).float().mean()
            all_accuracies.append(acc)
            if i % 100 == 0:
                val_loss, val_acc, cf = evaluate(model, val_dl, loss_fn, tokenizer, device)
                print(f"epoch {EPOCHS} step {i} train_loss: {loss.item():.3f}")
                print(f"epoch {EPOCHS} step {i} train_accuracy: {acc:.3f}")
                print(f"epoch {EPOCHS} step {i} val_loss: {val_loss:.3f}")
                print(f"epoch {EPOCHS} step {i} val_accuracy: {val_acc:.3f}")
                print(f"epoch {EPOCHS} step {i} val_CF: \n {cf}")
                model.train()

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict()
    }
    torch.save(checkpoint, MODEL_PATH / "jigsaw-toxic-comments" / f"distilbert_epochs{EPOCHS}_LR{LR}_model")
    import matplotlib.pyplot as plt
    loss_ema = [np.average(all_losses[i:i+10]) for i in range(len(all_losses)-10)]
    plt.plot(loss_ema)
    plt.show()

else:
    state_dict = torch.load(MODEL_PATH / "jigsaw-toxic-comments" / f"distilbert_epochs{EPOCHS}_LR{LR}_model")
    model.load_state_dict(state_dict['model'])
    opt.load_state_dict(state_dict['optimizer'])


val_loss, val_acc, cf = evaluate(model, val_dl, loss_fn, tokenizer, device)


print("finished")