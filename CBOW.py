import torch.nn as nn
import torch.nn.functional as F


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