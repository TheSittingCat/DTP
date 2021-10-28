import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import simpletransformers as sp
from simpletransformers.classification import ClassificationArgs,ClassificationModel
import sklearn as sk
import torch
#Load the model (Change the path to the path containing the model)
Model = ClassificationModel('roberta','/content/drive/MyDrive/Kaveh/2_Model/checkpoint-840-epoch-10')
test_df = pd.read_csv('/content/drive/MyDrive/en_val_reduced_2.csv') #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[2,3,4]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)
test_df = pd.read_csv('/content/drive/MyDrive/fr_test_reduced_2.csv') #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[2,3,4]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)
test_df = pd.read_csv('/content/drive/MyDrive/fr_valid_reduced_2.csv') #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[2,3,4]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)
test_df = pd.read_csv('/content/drive/MyDrive/de_valid_reduced_2.csv') #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[2,3,4]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)
test_df = pd.read_csv('/content/drive/MyDrive/de_test_reduced_2.csv') #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[3,4,5]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)
test_df = pd.read_csv('/content/drive/MyDrive/it_test_reduced_2.csv') #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[2,3,4]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)
test_df = pd.read_csv('/content/drive/MyDrive/fa_test_reduced_2.csv')  #The path to evaluation data pruned to n
test_df = test_df[test_df.columns[[2,3,4]]]
test_df.columns = ['text_a','text_b','labels']
test_df = test_df.sample(frac=1)
test_df['labels'] = pd.to_numeric(test_df['labels'], errors='coerce')
test_df = test_df.dropna(subset=['labels'])
results, outputs, wrong  = Model.eval_model(test_df, acc=sk.metrics.accuracy_score, f1 = sk.metrics.f1_score, verbose= False)
print(results)