"""
Following Simple tutorial for distilling BERT
By Pavel Gladkov
From: https://towardsdatascience.com/simple-tutorial-for-distilling-bert-99883894e90a
Date: 28.04.2020
"""

### BUILDING THE STUDENT MODEL
# For building vocabulary - using built-in functionality from the torchtext package
# These will help us translate words from the training dataset into word-indices

import torch
from torchtext import data
import wget

### Add some data
url = 'https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/news_category_train.csv'
wget.download(url, '/Users/maris/Documents/Aarhus University/Data science/data-science-exam/news_category_train.csv')

import pandas as pd
train_data = pd.read_csv("news_category_train.csv", encoding="utf8", sep = ",", quotechar='"', error_bad_lines=False)
###

def get_vocab(X):
    X_split = [t.split() for t in X]
    text_field = data.Field()
    text_field.build_vocab(X_split, max_size=10000)
    return text_field

def pad(seq, max_len):
    if len(seq) < max_len:
        seq = seq + ['<pad>'] * (max_len - len(seq))
    return seq[0:max_len]

def to_indexes(vocab, words):
    return [vocab.stoi[w] for w in words]

def to_dataset(x, y, y_real):
    torch_x = torch.tensor(x, dtype=torch.long)
    torch_y = torch.tensor(y, dtype=torch.float)
    torch_real_y = torch.tensor(y_real, dtype=torch.long)
    return TensorDataset(torch_x, torch_y, torch_real_y)

### MODEL
import torch
from torch import nn
from torch.autograd import Variable


class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, batch_size, device=None):
        super(SimpleLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = self.init_device(device)
        self.hidden = self.init_hidden()

    @staticmethod
    def init_device(device):
        if device is None:
            return torch.device('cuda')
        return device

    def init_hidden(self):
        return (Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)),
                Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).to(self.device)))

    def forward(self, text, text_lengths=None):
        self.hidden = self.init_hidden()
        x = self.embedding(text)
        x, self.hidden = self.rnn(x, self.hidden)
        hidden, cell = self.hidden
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        x = self.fc(hidden)

        return x

### TRAIN BASELINE
"""
For this particular baseline model, 
we set output_dim=1 because 
we have a binary classification, 
thus loss function is logloss.
PyTorch has the BCEWithLogitsLoss class, 
which combines sigmoid function and binary cross-entropy:
"""

def loss(self, output, bert_prob, real_label):
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(output, real_label.float())

# One epoch would be:
def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
    return optimizer, scheduler


def epoch_train_func(model, dataset, loss_func, batch_size):
    train_loss = 0
    train_sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=train_sampler,
                             batch_size=batch_size,
                             drop_last=True)
    model.train()
    optimizer, scheduler = get_optimizer(model)
    for i, (text, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Train')):
        text, bert_prob, real_label = to_device(text, bert_prob, real_label)
        model.zero_grad()
        output = model(text.t(), None).squeeze(1)
        loss = loss_func(output, bert_prob, real_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    return train_loss / len(data_loader)

# Evaluation after each epoch
def epoch_evaluate_func(model, eval_dataset, loss_func, batch_size):
    eval_sampler = SequentialSampler(eval_dataset)
    data_loader = DataLoader(eval_dataset, sampler=eval_sampler,
                             batch_size=batch_size,
                             drop_last=True)

    eval_loss = 0.0
    model.eval()
    for i, (text, bert_prob, real_label) in enumerate(tqdm(data_loader, desc='Val')):
        text, bert_prob, real_label = to_device(text, bert_prob, real_label)
        output = model(text.t(), None).squeeze(1)
        loss = loss_func(output, bert_prob, real_label)
        eval_loss += loss.item()

    return eval_loss / len(data_loader)

### FULL CODE FOR TRAINING
import os
import torch
from torch.utils.data import (TensorDataset, random_split,
                              RandomSampler, DataLoader,
                              SequentialSampler)
from torchtext import data
from tqdm import tqdm


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(text, bert_prob, real_label):
    text = text.to(device())
    bert_prob = bert_prob.to(device())
    real_label = real_label.to(device())
    return text, bert_prob, real_label


class LSTMBaseline(object):
    vocab_name = 'text_vocab.pt'
    weights_name = 'simple_lstm.pt'

    def __init__(self, settings):
        self.settings = settings
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device())

    def loss(self, output, bert_prob, real_label):
        return self.criterion(output, real_label.float())

    def model(self, text_field):
        model = SimpleLSTM(
            input_dim=len(text_field.vocab),
            embedding_dim=64,
            hidden_dim=128,
            output_dim=1,
            n_layers=1,
            bidirectional=True,
            dropout=0.5,
            batch_size=self.settings['train_batch_size'])
        return model

    def train(self, X, y, y_real, output_dir):
        max_len = self.settings['max_seq_length']
        text_field = get_vocab(X)

        X_split = [t.split() for t in X]
        X_pad = [pad(s, max_len) for s in tqdm(X_split, desc='pad')]
        X_index = [to_indexes(text_field.vocab, s) for s in tqdm(X_pad, desc='to index')]

        dataset = to_dataset(X_index, y, y_real)
        val_len = int(len(dataset) * 0.1)
        train_dataset, val_dataset = random_split(dataset, (len(dataset) - val_len, val_len))

        model = self.model(text_field)
        model.to(device())

        self.full_train(model, train_dataset, val_dataset, output_dir)
        torch.save(text_field, os.path.join(output_dir, self.vocab_name))

    def full_train(self, model, train_dataset, val_dataset, output_dir):
        train_settings = self.settings
        num_train_epochs = train_settings['num_train_epochs']
        best_eval_loss = 100000
        for epoch in range(num_train_epochs):
            train_loss = epoch_train_func(model, train_dataset, self.loss, self.settings['train_batch_size'])
            eval_loss = epoch_evaluate_func(model, val_dataset, self.loss, self.settings['eval_batch_size'])

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(output_dir, self.weights_name))

###
### DISTILLING
###

"""
We reuse almost the code entirely with only one modification— the loss function. For calculation cross-entropy loss I used CrossEntropyLoss, which combines softmax function and cross-entropy.
"""
def loss(self, output, bert_prob, real_label):
    a = 0.5
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()
    return a*criterion_ce(output, real_label) + (1-a)*criterion_mse(output, bert_prob)

class LSTMDistilled(LSTMBaseline):
    vocab_name = 'distil_text_vocab.pt'
    weights_name = 'distil_lstm.pt'

    def __init__(self, settings):
        super(LSTMDistilled, self).__init__(settings)
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.a = 0.5

    def loss(self, output, bert_prob, real_label):
        return self.a * self.criterion_ce(output, real_label) + (1 - self.a) * self.criterion_mse(output, bert_prob)

    def model(self, text_field):
        model = SimpleLSTM(
            input_dim=len(text_field.vocab),
            embedding_dim=64,
            hidden_dim=128,
            output_dim=2,
            n_layers=1,
            bidirectional=True,
            dropout=0.5,
            batch_size=self.settings['train_batch_size'])
        return model