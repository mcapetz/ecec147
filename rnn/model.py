import torch
from torch import nn
import torchtext
from torchtext.vocab import GloVe
from nltk import word_tokenize # very popular Text processing Library
import numpy as np

class Model(nn.Module):
    def __init__(self, embs_npa, dataset, num_layers=3, embedding_dim=100, lstm_size=100):
        super(Model, self).__init__()
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        n_vocab = len(dataset.uniq_words)
        self.num_embeddings=n_vocab

        # use pretrained embeddings
        self.vocab_size = embs_npa.shape[0]
        self.embedding_dim = embs_npa.shape[1]
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embs_npa).float(),
            freeze=False
            )
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
            
        embed = self.embedding(x) # this was alr here
        
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))