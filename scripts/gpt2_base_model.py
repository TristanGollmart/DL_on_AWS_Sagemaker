import torch.optim
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelWithLMHead
from datasets import Dataset
import pathlib
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor

from torch import autocast

# ------------ Relevant script parameters -------------------
ROOT_PATH = pathlib.Path("..")
DATA_PATH = ROOT_PATH / "data"
MODEL_PATH = ROOT_PATH / "models"
device = 'cpu'
MODEL = "openai-community/gpt2"  #'distilbert-base-multilingual-cased'
BATCH_SIZE = 4
EPOCHS = 30
MAX_DOC_LENGTH = 256
LR = 5e-5
RELOAD = "distilbert_generation_epochs30_LR5e-05_model"
N_SEQ_PER_COMMENT = 16
PADDING = "left"
# -----------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side=PADDING)
tokenizer.add_special_tokens({"additional_special_tokens": ["[neutral]", "[toxic]"]})

pretrained_model = AutoModelWithLMHead.from_pretrained(MODEL)
pretrained_model.resize_token_embeddings(len(tokenizer))
config = pretrained_model.config # information about model configuration -> use to attach correct classifier head


vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
#pretrained_model.generation_config = config
pretrained_model.generation_config.pad_token = tokenizer.pad_token
pretrained_model.generation_config.pad_token_id = tokenizer.pad_token_id

train_df = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "train.csv")[['comment_text', 'toxic']]#.sample(n=100*BATCH_SIZE)
df_neg = train_df[train_df['toxic']==0].sample(n=30*BATCH_SIZE)
df_pos = train_df[train_df['toxic']==1].sample(n=30*BATCH_SIZE)
train_df = pd.concat([df_neg, df_pos], axis=0)
test_df = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "test.csv").sample(n=2*BATCH_SIZE)
test_labels = pd.read_csv(DATA_PATH / "jigsaw-toxic-comments" / "test_labels.csv")

train_df, val_df = train_test_split(train_df, test_size=0.5, random_state=42, stratify=train_df['toxic'].values)


print(train_df['toxic'].value_counts())
print(val_df['toxic'].value_counts())
print(train_df.sample(4))

def tokenize(df):
    '''returns dict of enumerated keys and tokenized sentences as values'''
    tokenized_text = {i: tokenizer.encode(tokenizer.additional_special_tokens[df['toxic'].iloc[i]]) +
                         tokenizer(text,
              max_length=MAX_DOC_LENGTH,
              return_tensors='pt',
              truncation=True)['input_ids'].squeeze().tolist() for i, text in enumerate(df['comment_text'].values)}
    return tokenized_text

def group_into_equal_batches(data={}):
    results = {} # labels

    text_stream = []
    for text in data.values():
        text_stream.extend(text)
    tokenized_data = []
    labels = []
    # cut off remainder batch that does not fit into one MAX_DOC_LENGTH
    total_length = len(text_stream)
    total_length = (total_length // MAX_DOC_LENGTH) * MAX_DOC_LENGTH
    for i in range(0, total_length, MAX_DOC_LENGTH):
        tokenized_data.append(text_stream[i:i+MAX_DOC_LENGTH])
        labels.append(text_stream[(i+MAX_DOC_LENGTH) % len(text_stream)])

    results['inputs'] = tokenized_data
    results['labels'] = labels
    return results


class Train_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.X = pd.DataFrame(data['inputs'])
        self.y = pd.DataFrame(data['labels'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y.iloc[idx].values

class Model(nn.Module):
    def __init__(self, pretrained_model, vocab_size):
        super().__init__()
        self.base = pretrained_model #
        #self.head = nn.Linear(self.base.config.dim, vocab_size)

    def forward(self, **inputs):
        '''

        :param inputs: must contain inputs_ids, attention_mask
        :return:
        '''
        outputs = self.base(**inputs)
        return outputs
        #return self.head(out_seq)

    def generate_custom(self, input_text:str="", num_words:int=1, style:str="neutral"):
        # style ("neutral", "toxic") is the sentiment that generation is conditioned on.
        assert style in ["neutral", "toxic"], "style must be either of neutral or toxic"
        outputs = "[" + style + "]" + input_text
        encodings = tokenizer(outputs, max_length=MAX_DOC_LENGTH, return_tensors='pt', truncation=True)
        for i in range(num_words):
            logits = self.base(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask']).logits.squeeze()[
                -1]
            next_token = torch.argmax(logits, dim=-1)
            encodings["input_ids"] = torch.concat([encodings["input_ids"],
                                                   torch.Tensor([[next_token]]).type_as(encodings["input_ids"])] , dim=-1)
            encodings["attention_mask"] = torch.concat([encodings["attention_mask"], torch.Tensor([[1]])] , dim=-1)
            new_word = tokenizer.decode(next_token)
            outputs += new_word
        return outputs

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
        preds = np.array([])
        y_true= np.array([])

        for (i, batch) in enumerate(dataloader):
            input_ids, next_tokens = batch
            input_ids = input_ids.reshape(-1, MAX_DOC_LENGTH)
            next_tokens = next_tokens.reshape(-1)
            next_tokens.to(device)
            attention_mask = get_mask(input_ids, tokenizer)

            encodings = {"input_ids": input_ids,
                        "attention_mask": attention_mask,
                         "labels": input_ids
}

            outputs = model(**encodings)
            logits = outputs.logits.to(device)
            #logits.requires_grad = True

            losses.append(outputs.loss.item())

            probs = torch.softmax(logits, dim=-1)
            ypred = torch.argmax(probs, dim=-1)
            acc = (ypred == torch.roll(input_ids, shifts=-1, dims=-1)).float().mean()
            accuracies.append(acc)

            preds = np.concatenate([preds, np.array(ypred.int().squeeze().tolist()).reshape(-1)])
            y_true = np.concatenate([y_true, np.array(torch.roll(input_ids, shifts=-1, dims=-1).tolist()).reshape(-1)])

            if i>10:
                break
    cf = confusion_matrix(y_true, preds)
    return np.mean(losses), np.mean(accuracies), cf


model = Model(pretrained_model, vocab_size)
model.train()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LR)

tokens_train = tokenize(train_df)
train_data = group_into_equal_batches(tokens_train)
tokens_val = tokenize(val_df)
val_data = group_into_equal_batches(tokens_val)

train_dl = DataLoader(Train_Dataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(Train_Dataset(val_data), batch_size=BATCH_SIZE, shuffle=True)

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

if RELOAD is None:
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            input_length = 1 if model.base.config.is_encoder_decoder else input_ids.shape[1]
            #generated_tokens = outputs.sequences[:, input_length:]
            logits = outputs.logits.to(device)

            #logits.requires_grad = True
            loss = outputs.loss.to(device) #loss_fn(logits.squeeze(), next_tokens.long()).to(device)
            loss.backward()
            opt.step()

            all_losses.append(loss.item())

            probs = torch.softmax(logits, dim=-1)
            ypred = torch.argmax(probs, dim=-1)
            acc = (ypred == torch.roll(input_ids, shifts=-1, dims=-1)).float().mean()
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
else:
    state_dict = torch.load(MODEL_PATH / "jigsaw-toxic-comments" / RELOAD)
    model.load_state_dict(state_dict['model'])

generated_text_custom_neutral = model.generate_custom("I think this product ", num_words=50, style="neutral")
generated_text_custom_toxic = model.generate_custom("I think this product ", num_words=50, style="toxic")


style = "[toxic]"
input_ids = tokenizer.encode(style + "I think this product", return_tensors='pt')
generated_text = pretrained_model.generate(input_ids, max_length=50)
print(generated_text_custom_neutral)
print(generated_text_custom_toxic)
print(tokenizer.decode(generated_text.squeeze()))


if RELOAD is None:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict()
    }
    torch.save(checkpoint, MODEL_PATH / "jigsaw-toxic-comments" / f"distilbert_generation_epochs{EPOCHS}_LR{LR}_model")

