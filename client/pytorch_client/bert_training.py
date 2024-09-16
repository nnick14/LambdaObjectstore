#!/usr/bin/env python
# coding: utf-8

# In[81]:


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

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import GPT2Model


# In[82]:


# write dataloaders for different datasets from s3, disk and infinicache
# 1) options to choose datasets, will be using preprocessed data
# 2) seperate dataloaders for s3, disk and infinicache, should be batch dataloading

def s3_dataloader():
    ## To do
    pass

def disk_dataloader(data, datatype, batch_size):
    ## To be updated
    #batch_size = 16  #16 or 32 is the right value
    input_ids,attention_masks,labels = data_to_embeddings(data, datatype)
    # create dataset
    dataset_bert = TensorDataset(input_ids,attention_masks,labels)
    # sampler
    custom_sampler = RandomSampler(dataset_bert)
    #load dataset
    dataloader = DataLoader(dataset_bert,
                              sampler = custom_sampler,
                              batch_size = batch_size)
    return dataloader

def infinicache_dataloader():
    # To do
    pass

# 3)check dataloading times for all datasets using different dataloaders
def check_dataloading_time(dataset_type, dataloader_type):
    pass


# In[83]:


# initialize model/architecture for bert and gpt2 (bagof words, tf-idf, wordvec is optional)
def data_to_embeddings(data, datatype):
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenize data
    if datatype =="train":
        data_values = data[data.data_type == 'train'].text.values
        data_labels = data[data.data_type == 'train'].label.values
    elif datatype =="valid":
        data_values = data[data.data_type == 'valid'].text.values
        data_labels = data[data.data_type == 'valid'].label.values
    else:
        raise Exception("Data Type do not match!!!!")
    encoded_data = tokenizer.batch_encode_plus(data_values,
                                                add_special_tokens = True,
                                                return_attention_mask = True,
                                                pad_to_max_length = True,
                                                max_length = 150,
                                                return_tensors = 'pt')
    #encode data
    input_ids = torch.tensor(encoded_data['input_ids'])
    attention_masks = torch.tensor(encoded_data['attention_mask'])
    labels = torch.tensor(data_labels)
    print("Sucessful embeddings!!! for {}".format(datatype))
    return input_ids, attention_masks, labels 

#initialize model
def initialize_model(labels):
    #load pre-trained BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels = len(labels),
                                                          output_attentions = False,
                                                          output_hidden_states = False) 
    # choose device for our model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

# model evaluation
def evaluate_model(model, dataloader_valid):
    #put the model in evaluation mode which disables the dropout layer 
    model.eval()
    
    #tracking variables
    total_loss_value = 0
    predictions, value_accuracy = [], []
    
    # choose device for our model # there was constant issue for this
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for step,batch in enumerate(dataloader_valid):
        #load batch into GPU
        batch = tuple(b.to(device) for b in batch)
        #define inputs
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2]}

        #compute logits
        with torch.no_grad():        
            outputs = model(**inputs)
        
        #compute loss
        loss = outputs[0]
        logits = outputs[1]
        total_loss_value += loss.item()

        #compute accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        value_accuracy.append(label_ids)
    
    #compute average loss
    total_loss_value = total_loss_value/len(dataloader_valid) 
    predictions = np.concatenate(predictions, axis=0)
    value_accuracy = np.concatenate(value_accuracy, axis=0)
    
    return total_loss_value, predictions, value_accuracy

#f1 score, accuracy score
def f1_score_func(predictions, labels):
    """F1 Score used for calculating accuracy.
    """
    preds_flatten = np.argmax(predictions, axis=1).flatten()
    labels_flatten = labels.flatten()
    return f1_score(labels_flatten, preds_flatten, average = 'weighted')

def set_seed(seed_value):
    """Set seed value for reproducibility.
    """
    print("Seed Value is set!!!")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    


# In[87]:


# training in epochs and batchs
def train(epochs, model, dataloader_train,dataloader_valid, evaluation=False):
    """Train BertClassifier model with our dataset.
    """
    print("Training has Started!!!")
    #load optimizer
    optimizer = AdamW(model.parameters(),
                     lr = 1e-5,
                     eps = 1e-8)
    # load scheduler 
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps = 0,
                                               num_training_steps = len(dataloader_train)*epochs)
    # choose device for our model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(1, epochs+1):
        #set model in train mode
        model.train()
        # measure the elapsed time of each epoch
        total_time_epoch, total_time_batch = time.time(), time.time()
        # rest tracking variable at every epoch
        total_train_loss, batch_loss, batch_counts= 0,0,0
        print("Epoch:{}".format(epoch))
        
        #print('Epochs: {}'.format(loss_train_avg)
        for step,batch in enumerate(dataloader_train):
            # add the batch counts
            batch_counts+=1
            #set gradient to 0 for every batch
            model.zero_grad()
            #load batch into GPU
            batch = tuple(b.to(device) for b in batch)
            #define inputs for our bert model
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            
            # perform forward pass, this gives logits as output
            outputs = model(**inputs)
            # compute loss and accumulate loss values
            loss = outputs[0] #output.loss
            batch_loss+=loss.item()
            total_train_loss +=loss.item()

            # do backward pass to get gradients
            loss.backward()
            
            #clip the norm of the gradients to 1.0 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #update optimizer
            optimizer.step()
            #update scheduler
            scheduler.step()
            
            if (batch_counts == 24):
                # total batch counts
                #print('Batch counts: {}'.format(batch_counts))
                # batch time elapsed
                batch_time_elapsed = time.time() - total_time_batch
                #print('Batch Training Time: {}'.format(batch_time_elapsed))
                # batch loss
                loss_batch_avg = batch_loss/len(batch)
                #print('Batch training loss: {}'.format(loss_batch_avg))
                # reinitilize batch related values
                total_time_batch =time.time()
                batch_loss, batch_counts = 0,0
            
        #print avergae training loss over each epoch
        loss_train_avg = total_train_loss/len(dataloader_train)
        print('Training loss: {}'.format(loss_train_avg))
        if evaluation==True:  
            val_loss, predictions, true_vals = evaluate_model(model, dataloader_valid)
            #f1 score
            val_f1 = f1_score_func(predictions, true_vals)
            print("Validation loss:{}".format(val_loss))
            print("F1 Score (weighted:{}".format(val_f1))
            
        # Print performance over the training data at each epoch
        epoch_time_elapsed = time.time() - total_time_epoch
        print('Training time : {}'.format(epoch_time_elapsed)) 
        


# In[88]:


def data_preprocessing():
    data = pd.read_csv('./simle-sentiment-analysis-final.csv')
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


# In[89]:



# for bert
batch_size = 24
modeltype = "bert"
data= data_preprocessing()

## data loader
datatype, batch_size, modeltype, labels ="train", 24, "bert", data.label.unique() 
dataloader_train = disk_dataloader(data, datatype, batch_size)

datatype, batch_size, modeltype, labels ="valid", 24, "bert", data.label.unique()
dataloader_valid = disk_dataloader(data, datatype, batch_size)
## initialize model
model = initialize_model(labels=data.label.unique())

##
train(epochs, model, dataloader_train,dataloader_valid, evaluation=False)


# In[ ]:





# ##### For disk
# 
# 
