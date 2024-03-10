import torch.optim
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import pathlib
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor

from torch import autocast


ROOT_PATH = pathlib.Path("..")
DATA_PATH = ROOT_PATH / "data"
MODEL_PATH = ROOT_PATH / "models"
device = 'cpu'
MODEL = "openai-community/gpt2"  #'distilbert-base-multilingual-cased'
BATCH_SIZE = 4
EPOCHS = 10
MAX_DOC_LENGTH = 256
LR = 5e-5
RELOAD = False
N_SEQ_PER_COMMENT = 16
PADDING = "left"

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side=PADDING)
pretrained_model = AutoModelForCausalLM.from_pretrained(MODEL)
config = pretrained_model.config # information about model configuration -> use to attach correct classifier head


vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

pretrained_model.generation_config.pad_token = tokenizer.pad_token
pretrained_model.generation_config.pad_token_id = tokenizer.pad_token_id

train_df = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "train.csv")[['comment_text', 'toxic']]
df_neg = train_df[train_df['toxic']==0].sample(n=BATCH_SIZE)
df_pos = train_df[train_df['toxic']==1].sample(n=BATCH_SIZE)
train_df = pd.concat([df_neg, df_pos], axis=0)
test_df = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "test.csv").sample(n=2*BATCH_SIZE)
test_labels = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "test_labels.csv")

train_df, val_df = train_test_split(train_df, test_size=0.5, random_state=42, stratify=train_df['toxic'].values)
print(train_df['toxic'].value_counts())
print(val_df['toxic'].value_counts())
print(train_df.sample(4))

class Train_Dataset(Dataset):
    def __init__(self, df, tokenizer, chunk_size, padding: str = 'right'):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.padding = padding
        encodings =[tokenizer(comment,
                              max_length=MAX_DOC_LENGTH,
                              return_tensors='pt',
                              padding="max_length", truncation=True)["input_ids"].squeeze() for comment in df['comment_text'].values]

        for i in range(len(encodings)):
            comment_len = torch.sum(torch.not_equal(encodings[i], self.pad_id).type(torch.int))
            if comment_len < chunk_size:
                if self.padding == 'left':
                    #old = encodings[i][-chunk_size:].clone()
                    encodings[i][-chunk_size:] = torch.concatenate([encodings[i][-comment_len:] , encodings[(i + 1) % len(encodings)][
                                                                -chunk_size + comment_len:]])
                    #new = encodings[i][-chunk_size:]
                elif self.padding == 'right':
                    encodings[i][comment_len+1: chunk_size] = encodings[(i+1) % len(encodings)][:chunk_size-comment_len-1]
                else:
                    raise ValueError("padding must be either <left> or <right>.")
        self.X = encodings #["input_ids"]
        #self.y = df['toxic'].values
        self.chunk_size = chunk_size


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        #comment_len = self.X[idx].size()[-1]
        #repeated_comments = self.X[idx].repeat(min(comment_len, self.chunk_size), 1)

        # return chunks of sentence, masking varying number of tokens at the end of sentence
        comment_len = torch.sum(torch.not_equal(self.X[idx], self.pad_id).type(torch.int))
        if self.padding == 'left':
            repeated_comments = torch.stack([
                                    torch.concat([
                                    torch.Tensor([self.pad_id]*(j+1)), self.X[idx][:-j-1]]) for j in range(self.chunk_size)])
            labels = torch.zeros(repeated_comments.shape[0])
            for i in range(self.chunk_size):
                repeated_comments[i, :-comment_len+i+1] = self.pad_id
                labels[i] = self.X[idx][-i-1]
        elif self.padding == 'right':
            repeated_comments = self.X[idx].repeat(self.chunk_size, 1)
            labels = torch.zeros(repeated_comments.shape[0])
            for i in range(self.chunk_size):
                repeated_comments[i, comment_len-i-1:] = self.pad_id
                labels[i] = self.X[idx][comment_len-i-1]
        else:
            raise ValueError("padding must be either <left> or <right>.")
        return repeated_comments.long(), labels

class Model(nn.Module):
    def __init__(self, pretrained_model, vocab_size):
        super().__init__()
        self.base = pretrained_model #
        #self.head = nn.Linear(self.base.config.dim, vocab_size)

    def forward(self, input_ids, attention_mask):
        out_seq = self.base(input_ids, attention_mask).last_hidden_state
        return out_seq
        #return self.head(out_seq)

    def generate(self, **inputs):
        return self.base.generate(**inputs)

def get_mask(input_ids, tokenizer):
    pad_id = tokenizer.encode(tokenizer.pad_token)[0]
    mask = torch.ones_like(input_ids)
    mask[input_ids==pad_id] = 0
    return mask

def evaluate(model, dataloader, loss_fn, tokenizer, device):
    with torch.no_grad():
        model.eval()
        losses = []
        accuracies = []
        preds = []
        y_true= []

        for (i, batch) in enumerate(dataloader):
            input_ids, next_tokens = batch
            input_ids = input_ids.reshape(-1, MAX_DOC_LENGTH)
            next_tokens = next_tokens.reshape(-1)
            next_tokens.to(device)
            attention_mask = get_mask(input_ids, tokenizer)

            encodings = {"input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": 1,
                        "return_dict_in_generate": True,
                        "output_scores": True}

            outputs = model.generate(**encodings)
            logits = outputs.scores[0].to(device)
            logits.requires_grad = True

            losses.append(loss_fn(logits.squeeze(), next_tokens.long()).item())

            probs = torch.softmax(logits, dim=-1)
            ypred = torch.argmax(probs, dim=-1)
            acc = (ypred == next_tokens).float().mean()
            accuracies.append(acc)

            preds += ypred.int().squeeze().tolist()
            y_true += next_tokens.tolist()

            if i>10:
                break
    cf = confusion_matrix(y_true, preds)
    return np.mean(losses), np.mean(accuracies), cf

train_dl = DataLoader(Train_Dataset(train_df, tokenizer, chunk_size=N_SEQ_PER_COMMENT, padding=PADDING), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(Train_Dataset(val_df, tokenizer, chunk_size=N_SEQ_PER_COMMENT, padding=PADDING), batch_size=BATCH_SIZE, shuffle=True)


model = Model(pretrained_model, vocab_size)
model.train()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)



# example generation
'''
comment = ["What is the first name of Obama?"]
encodings = tokenizer(comment, max_length=MAX_DOC_LENGTH,
                              return_tensors='pt',
                              padding="max_length", truncation=True)
#comment_len = torch.sum(torch.not_equal(comment, tokenizer.pad_token_id).type(torch.int))
outputs = model.generate(**encodings, max_new_tokens=20, return_dict_in_generate=True, output_scores=True, num_beams=4, repetition_penalty=0.7)
print(tokenizer.decode(outputs.sequences[0][MAX_DOC_LENGTH:]))
'''


all_losses = []
all_accuracies = []
for epochs in range(EPOCHS):
    model.to(device)
    model.train()
    for (i, batch) in enumerate(train_dl):
        input_ids, next_tokens = batch
        input_ids = input_ids.reshape(-1, MAX_DOC_LENGTH)
        next_tokens = next_tokens.reshape(-1)
        #x = [self.X[idx].squeeze()[:comment_len - i] for i in range(1, 1+min(comment_len-1, self.chunk_size))]
        #y = torch.Tensor([self.X[idx].squeeze()[comment_len - i] for i in range(1, 1 + min(comment_len-1, self.chunk_size))])

        # x.to(device)
        next_tokens.to(device)
        attention_mask = get_mask(input_ids, tokenizer)
        #input_ids = encodings['input_ids'].to(device)
        #attention_mask = encodings['attention_mask'].to(device)
        #logits = model(input_ids, attention_mask).to(device)

        opt.zero_grad()
        #with torch.autocast(device_type=device):
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
        #input_length = 1 if model.base.config.is_encoder_decoder else input_ids.shape[1]
        #generated_tokens = outputs.sequences[:, input_length:]
        logits = outputs.logits.to(device)

        logits.requires_grad = True
        loss = outputs.loss.to(device) #loss_fn(logits.squeeze(), next_tokens.long()).to(device)

        loss.backward()
        opt.step()

        all_losses.append(loss.item())

        probs = torch.softmax(logits, dim=-1)
        ypred = torch.argmax(probs, dim=-1)
        acc = (ypred == next_tokens).float().mean()
        all_accuracies.append(acc)
        if i % 100 == 0:
            opt.zero_grad()
            val_loss, val_acc, cf = evaluate(model, val_dl, loss_fn, tokenizer, device)
            print(f"epoch {epochs} step {i} train_loss: {loss.item():.3f}")
            print(f"epoch {epochs} step {i} train_accuracy: {acc:.3f}")
            print(f"epoch {epochs} step {i} val_loss: {val_loss:.3f}")
            print(f"epoch {epochs} step {i} val_accuracy: {val_acc:.3f}")
            print(f"epoch {epochs} step {i} val_CF: \n {cf}")
            model.train()

checkpoint = {
    'model': model.state_dict(),
    'optimizer': opt.state_dict()
}
torch.save(checkpoint, MODEL_PATH / "jigsaw-toxic-comments" / f"distilbert_generation_epochs{EPOCHS}_LR{LR}_model")

#ausal_mask = torch.zeros((seq_len, seq_len))
#mask = attention_mask & causal_mask