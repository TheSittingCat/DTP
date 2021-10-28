import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import simpletransformers as sp
from simpletransformers.classification import ClassificationArgs,ClassificationModel
import sklearn as sk
import torch
train_df = pd.read_csv('/content/drive/MyDrive/en_train_reduced_2.csv') #The adress to english training set reduced to level 2
train_df = train_df[train_df.columns[[2,3,4]]]
train_df.columns = ['text_a','text_b','labels']
train_df['labels'] = pd.to_numeric(train_df['labels'], errors='coerce')
train_df = train_df.dropna(subset=['labels'])
#The hyperparameters used for all our models. feel free to change them.
Model_arguments = ClassificationArgs()
Model_arguments.regression = False
Model_arguments.num_train_epochs = 15
Model_arguments.reprocess_input_data = True
Model_arguments.train_batch_size = 64
Model_arguments.learning_rate = 3e-5
Model_arguments.no_cache = True
Model_arguments.overwrite_output_dir = True
Model_arguments.no_save = False
Model = ClassificationModel('roberta','xlm-roberta-base',args=Model_arguments,use_cuda=True) #Model initialization
Model.train_model(train_df,acc=lambda truth, predictions: accuracy_score(truth, [round(p) for p in predictions])) #This will train the model
