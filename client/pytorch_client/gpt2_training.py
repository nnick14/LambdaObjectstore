#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Import libraries and packages
import time
import random
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm
from transformers import BertModel



# Datasets  (BERT and GPT-2)
# GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
labels = {
    "business": 0,
    "entertainment": 1,
    "sport": 2,
    "tech": 3,
    "politics": 4
         }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in df['text']]
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
    
#Create the GPT2Classfier class
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        """
        Args: input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output
    
def train(epochs, model, train_data, val_data, learning_rate= 1e-5):
    ## make dataset
    train, val = Dataset(train_data), Dataset(val_data)
    ## data loader
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            
            model.zero_grad()

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")



def data_preprocessing():
    data = pd.read_csv('./bbc-text.csv')
    label_dict = {}
    possible_labels = data.category.unique()
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    data['label'] = data.category.replace(label_dict)
    X = data.text.values
    y = data.label.values
    #train test split
    X_train, X_val, y_train, y_val = train_test_split(data.index.values, 
                                                   data.label.values,
                                                   test_size = 0.15,
                                                   random_state = 17,
                                                   stratify = data.label.values)
    
    data['data_type'] = ['not_set'] * data.shape[0]
    data.loc[X_train, 'data_type'] = 'train'
    data.loc[X_val, 'data_type'] = 'valid'
    return data



# df = pd.read_csv('./bbc-text.csv')
# np.random.seed(112)
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=35),
#                                   [int(0.8*len(df)), int(0.9*len(df))])

## data preprocessing 
data=data_preprocessing() 

## data loader
train_data, val_data = np.split(data.sample(frac=1, random_state=35),
                                      [int(0.8*len(data))])
## model
epochs = 3
model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")

## train
train(epochs, model, train_data, val_data)
## predictions


