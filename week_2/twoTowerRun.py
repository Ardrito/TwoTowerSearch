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
from CBOW import CBOW
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

#marco_test = np.load("marco_test.npy",allow_pickle=True)
marco_val = np.load("marco_val.npy",allow_pickle=True)
marco_train = np.load("marco_train.npy",allow_pickle=True)

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = CBOW()#.to(device)
embedder.load_state_dict(torch.load("checkpoints/best.pt", weights_only=True))
embedder.eval()

class twoTowerDataSet(Dataset):
    def __init__(self, marco_splt, embedder, device):
        self.data = marco_splt
        self.len = self.data.shape[0]
        self.embedder = embedder.to(device)
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        randIdx = random.randint(0, self.len - 1)
        row = self.data[idx]
        query_ids = torch.tensor(row[1], dtype=torch.long, device=self.device)
        pos_idx = row[2].index(1)
        pos_ids = torch.tensor(row[pos_idx + 2], dtype=torch.long, device=self.device)
        neg_ids = torch.tensor(self.data[randIdx][3], dtype=torch.long, device=self.device)

        with torch.no_grad():  # No grad for embedder
            query = self.embedder.embed(query_ids).mean(dim=0)
            pos = self.embedder.embed(pos_ids).mean(dim=0)
            neg = self.embedder.embed(neg_ids).mean(dim=0)

        return query, pos, neg


class twoTowerModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32)
            )

    def forward(self,X):
        out = self.seq(X)

        return(out)
    
def contrastiveLoss(query, pos, neg, m=0.6):
    # query = query[0]
    # pos = pos[0]
    # neg = neg[0]
    sim_pos = F.cosine_similarity(query, pos)
    sim_neg = F.cosine_similarity(query, neg)
    # print ("pos",torch.dot(query,pos))
    # print (torch.linalg.norm(query),torch.linalg.norm(pos))
    # print ("neg", cosine_sim_neg)
    return torch.clamp(m - sim_pos + sim_neg, min=0.0).mean()


epoch_val_loss_history = []
epoch_train_loss_history = []

def load_model(model, optimizer, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resumed from epoch {epoch}, loss {loss}")
        return model, optimizer, epoch, loss
    else:
        print("No checkpoint found, starting from scratch")
        return model, optimizer, 0, float("inf")


def train(batchSize=1024, numEpochs= 20, lr=1e-3 ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = CBOW()
    embedder.load_state_dict(torch.load("checkpoints/best.pt", weights_only=True))
    embedder.eval().to(device)

    train_dataset = twoTowerDataSet(marco_train, embedder, device)
    val_dataset = twoTowerDataSet(marco_val, embedder, device)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)

    passage_model = twoTowerModel().to(device)
    query_model = twoTowerModel().to(device)

    optimizer = optim.Adam(list(passage_model.parameters()) + list(query_model.parameters()),lr=lr)

    query_model, optimizer, epoch_counter, best_loss = load_model(query_model, optimizer, 'checkpoints/bestQuery.pt', device)
    passage_model, optimizer, epoch_counter, best_loss = load_model(passage_model, optimizer, 'checkpoints/bestPassage.pt', device)


    best_loss = float("inf")

    epoch_counter = 0

    for epoch in range(numEpochs):
        #epoch_counter += 1
        epoch_train_loss = 0.0

        passage_model.train()
        query_model.train()
        
        for query, pos, neg in tqdm(train_loader):
            query = query.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()

            embedded_query = query_model(query)
            embedded_pos = passage_model(pos)
            embedded_neg = passage_model(neg)

            loss = contrastiveLoss(embedded_query, embedded_pos, embedded_neg)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_val_loss = 0.0

        passage_model.eval()
        query_model.eval()

        with torch.no_grad():
            for query, pos, neg in tqdm(val_loader):
                query = query.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                embedded_query = query_model(query)
                embedded_pos = passage_model(pos)
                embedded_neg = passage_model(neg)

                loss = contrastiveLoss(embedded_query, embedded_pos, embedded_neg)

                epoch_val_loss += loss.item()
        avg_train_loss = epoch_train_loss/len(train_loader)
        avg_val_loss = epoch_val_loss/len(val_loader)

        print(f"\nEpoch {epoch+1}/{numEpochs} — " f"Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}\n\n")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            #torch.save(query_model.state_dict(), f'checkpoints/bestQuery.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': query_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,}, f'checkpoints/bestQuery.pt')
            #torch.save(passage_model.state_dict(), f'checkpoints/bestPassage.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': passage_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,}, f'checkpoints/bestPassage.pt')
            print(f"Model improved. Saved.\n\n")


train()

    


