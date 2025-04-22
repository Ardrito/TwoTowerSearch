from datasets import load_dataset
import random
import re
from tqdm import tqdm
import pickle
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open("tokenized_marco/tokenized_marco_text","rb") as file:
    tokenized_marco_text = pickle.load(file)

class embed_train_dataset(Dataset):
    def __init__(self, words, window=2):
        self.data = words
        self.window = window
    
    def __len__(self):
        return len(self.data)-4
    
    def __getitem__(self, idx):
        idx = idx+self.window    
        sent = self.data[max(0,idx-self.window):min(idx+self.window+1,len(self.data))]    
        if len(sent) > 1:
            rand_idx = random.randint(0,len(sent)-1)
            target = sent[rand_idx]
            del sent[rand_idx]
            #print (sent)
            tokenized = torch.tensor(sent)
            #print (tokenized)
            
            return tokenized, torch.tensor(target)
        

dataset = embed_train_dataset(tokenized_marco_text)
dataloader = DataLoader(dataset, batch_size=1,shuffle=True)

for data in dataloader:
    print (data)
    break

class CBOW(nn.Module):
    def __init__(self, vocab_size = 76289, embedding_dim = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embedding_dim)   
        self.lin = nn.Linear(embedding_dim,vocab_size)

    def forward(self,inputs):
        # print (inputs)
        # print(inputs.shape)
        #print (inputs.argmax())
        embs = self.embed(inputs)
        embs = embs.mean(dim=1)
        out = self.lin(embs)
        probs = F.log_softmax(out,dim=1)
        return probs

def train_loop():
    number_epochs = 100

    #train_wiki, val_wiki = train_test_split(words)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    dataset = embed_train_dataset(tokenized_marco_text)
    dataloader = DataLoader(dataset, batch_size=2048,shuffle=True)

    
    model = CBOW().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    checkpoint_path = "checkpoints/best.pt"
    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float("inf"))


    best_loss = 100000000000000.0
    for epoch in range(number_epochs):
        model.train()
        epoch_loss = 0.0
        for X,Y in tqdm(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = F.cross_entropy(pred,Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            #print (loss.item())
        epoch_loss = epoch_loss/len(dataloader)
        print(f"Epoch: {epoch}/{number_epochs}, loss: {epoch_loss} ")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'checkpoints/best.pt')
            print(f"Model improved. Saved.")

train_loop()