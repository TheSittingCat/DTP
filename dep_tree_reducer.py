import os
import random
import shutil
from datetime import datetime
from IPython.display import clear_output
 
import torch, torchtext
import stanza
from official.nlp import optimization 
from transformers import AutoTokenizer, XLMRobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
 
import matplotlib.pyplot as plt
 
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

 
header_names = ['target_word', 'PoS', 'start-char-index_1', 'end-char-index_1', 'start-char-index_2', 'end-char-index_2', 'example_1', 'example_2', 'label']
 
en_train_data = pd.read_csv('/content/xlwic_datasets/wic_english/train_en.txt', sep='\t', header=None, names=header_names)
fr_train_data = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/french_fr/fr_train.txt', sep='\t', header=None, names=header_names)
de_train_data = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/german_de/de_train.txt', sep='\t', header=None, names=header_names)
it_train_data = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/italian_it/it_train.txt', sep='\t', header=None, names=header_names)
 
en_test_data  = pd.read_csv('/content/xlwic_datasets/wic_english/valid_en.txt', sep='\t', header=None, names=header_names)
 
fr_test_data  = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/french_fr/fr_test_data.txt', sep='\t', header=None, names=header_names)
fr_test_gold  = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/french_fr/fr_test_gold.txt', sep='\t', header=None, names=[header_names[-1]])
 
de_test_data  = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/german_de/de_test_data.txt', sep='\t', header=None, names=header_names)
de_test_gold  = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/german_de/de_test_gold.txt', sep='\t', header=None, names=[header_names[-1]])
 
it_test_data  = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/italian_it/it_test_data.txt', sep='\t', header=None, names=header_names)
it_test_gold  = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/italian_it/it_test_gold.txt', sep='\t', header=None, names=[header_names[-1]])
 
fa_test_data  = pd.read_csv('/content/xlwic_datasets/xlwic_wn/farsi_fa/fa_test_data.txt', sep='\t', header=None, names=header_names)
fa_test_gold  = pd.read_csv('/content/xlwic_datasets/xlwic_wn/farsi_fa/fa_test_gold.txt', sep='\t', header=None, names=[header_names[-1]])
 
 
fr_val_data   = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/french_fr/fr_valid.txt', sep='\t', header=None, names=header_names)
de_val_data   = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/german_de/de_valid.txt', sep='\t', header=None, names=header_names)
it_val_data   = pd.read_csv('/content/xlwic_datasets/xlwic_wikt/italian_it/it_valid.txt', sep='\t', header=None, names=header_names)
fa_val_data   = pd.read_csv('/content/xlwic_datasets/xlwic_wn/farsi_fa/fa_valid.txt', sep='\t', header=None, names=header_names)
 
 
 
fa_test_data = fa_test_data.drop('label', axis=1).join(fa_test_gold)
fr_test_data = fr_test_data.drop('label', axis=1).join(fr_test_gold)
de_test_data = de_test_data.drop('label', axis=1).join(de_test_gold)
it_test_data = it_test_data.drop('label', axis=1).join(it_test_gold)

def find_parent_id(deptree, id):
  parent_id = -1
  for word in deptree[0].words:
    if word.id == id:
      return word.head

import functools
def dep_prune(lang, data, maxdepth):
  stanza.download(lang)
  nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma,depparse')
 
  err_list = []
  df = pd.DataFrame(columns = ['index', 'example_1', 'example_2', 'label'])
 
  for i, row in data[['target_word', 'example_1', 'example_2', 'label']].iterrows():
    target_lem = nlp(row.target_word.lower()).sentences[0].words[0].lemma
    e1_target_id = -1
    e2_target_id = -1
    try : 
      e1 = nlp(row.example_1.lower())
      e2 = nlp(row.example_2.lower())
    except : 
      continue
    for word in e1.sentences[0].words:
      if word.lemma == target_lem:
        e1_target_id = word.id
    for word in e2.sentences[0].words:
      if word.lemma == target_lem:
        e2_target_id = word.id
    if e1_target_id == -1 or e2_target_id == -1: 
      err_list.append(i)
      continue
 
    q1 = [e1_target_id]
    depth = 0
    e1_reduced = set()
    while depth < maxdepth and q1:
      pid = q1.pop()
      for word in e1.sentences[0].words:
        if word.id == pid:
          e1_reduced.add(word)
          q1.append(word.head)
        if word.head == pid:
          if word.upos != 'PUNCT':
            e1_reduced.add(word)
      depth += 1
 
    q2 = [e2_target_id]
    depth = 0
    e2_reduced = set()
    while depth < maxdepth and q2:
      pid = q2.pop()
      for word in e2.sentences[0].words:
        if word.id == pid:
          e2_reduced.add(word)
          q2.append(word.head)
        if word.head == pid:
          if word.upos != 'PUNCT':
            e2_reduced.add(word)
      depth += 1
 
    o1 = functools.reduce(lambda y, x: y + ' ' + x.text, sorted(e1_reduced, key=lambda x: x.id), '')
    o2 = functools.reduce(lambda y, x: y + ' ' + x.text, sorted(e2_reduced, key=lambda x: x.id), '')
 
    df = df.append({'index': i, 'example_1': o1, 'example_2': o2, 'label': row.label}, ignore_index=True)
 
  return df

#@title Word Levels 

def word_levels(lang, sent, target_word):
  stanza.download(lang)
  nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma,depparse')
 
  levels = {}
 
  target_lem = nlp(target_word.lower()).sentences[0].words[0].lemma
  e1_target_id = -1
  e2_target_id = -1
  try : 
    e1 = nlp(sent.lower())
  except : 
    pass
  for word in e1.sentences[0].words:
    if word.lemma == target_lem:
      e1_target_id = word.id
  if e1_target_id == -1: 
    raise Exception
    pass

  seen = []
  q1 = [(0, e1_target_id)]
  depth = 0
  while q1:
    depth, pid = q1.pop()
    seen.append(pid)
    for word in e1.sentences[0].words:
      if word.id == pid:
        if word.head not in seen:
          if word.text not in levels:
            levels[word.text] = []
          levels[word.text].append(depth)
          q1.append((depth + 1, word.head))
      if word.head == pid:
        if word.id not in seen:
          if word.text not in levels:
            levels[word.text] = []
          levels[word.text].append(depth+1)
          if word.id not in seen:
            q1.append((depth + 1, word.id))
 
  return levels

#@title RUN THIS
 
df = dep_prune('fa', fa_test_data, 3)
df.to_csv('drive/MyDrive/fa_test_reduced_3.csv')
df = dep_prune('fa', fa_test_data, 2)
df.to_csv('drive/MyDrive/fa_test_reduced_2.csv')
df = dep_prune('fa', fa_test_data, 1)
df.to_csv('drive/MyDrive/fa_test_reduced_1.csv')

df = dep_prune('fa', fa_val_data, 3)
df.to_csv('drive/MyDrive/fa_val_reduced_3.csv')
df = dep_prune('fa', fa_val_data, 2)
df.to_csv('drive/MyDrive/fa_val_reduced_2.csv')
df = dep_prune('fa', fa_val_data, 1)
df.to_csv('drive/MyDrive/fa_val_reduced_1.csv')

#df = dep_prune('it', it_test_data, 3)
#df.to_csv('drive/MyDrive/it_test_reduced_3.csv')
#df = dep_prune('it', it_test_data, 1)
#df.to_csv('drive/MyDrive/it_test_reduced_1.csv')
#df = dep_prune('it', it_val_data, 3)
#df.to_csv('drive/MyDrive/it_val_reduced_3.csv')
#df = dep_prune('it', it_val_data, 1)
#df.to_csv('drive/MyDrive/it_val_reduced_1.csv')
#df = dep_prune('de', de_test_data, 3)
#df.to_csv('drive/MyDrive/de_test_reduced_3.csv')
#df = dep_prune('de', de_test_data, 2)
#df.to_csv('drive/MyDrive/de_test_reduced_2.csv')
#df = dep_prune('de', de_test_data, 1)
#df.to_csv('drive/MyDrive/de_test_reduced_1.csv')
#df = dep_prune('de', de_val_data, 3)
#df.to_csv('drive/MyDrive/de_val_reduced_3.csv')
#df = dep_prune('de', de_val_data, 1)
#df.to_csv('drive/MyDrive/de_val_reduced_1.csv')
df = dep_prune('en', en_test_data, 1)
df.to_csv('drive/MyDrive/en_val_reduced_1.csv')
df = dep_prune('en', en_test_data, 2)
df.to_csv('drive/MyDrive/en_val_reduced_2.csv')
df = dep_prune('en', en_test_data, 3)
df.to_csv('drive/MyDrive/en_val_reduced_3.csv')

# df = dep_prune('it', it_val_data, 2)
# df.to_csv('drive/MyDrive/it_valid_reduced_2.csv')
# df = dep_prune('fa', fa_val_data, 2)
# df.to_csv('drive/MyDrive/fa_valid_reduced_2.csv')
# df = dep_prune('it', it_test_data, 2)
# df.to_csv('drive/MyDrive/it_test_reduced_2.csv')


# df = dep_prune('de', de_val_data, 2)
# df.to_csv('drive/MyDrive/de_valid_reduced_2.csv')
# df = dep_prune('de', de_test_data, 2)
df.to_csv('drive/MyDrive/de_test_reduced_2.csv')

df.to_csv('drive/MyDrive/fr_test_reduced_3.csv')

#df = dep_prune('fr', fr_val_data, 3)
#df.to_csv('drive/MyDrive/fr_valid_reduced_3.csv')
#df = dep_prune('fr', fr_val_data, 2)
#df.to_csv('drive/MyDrive/fr_valid_reduced_2.csv')
#df = dep_prune('fr', fr_val_data, 1)
#df.to_csv('drive/MyDrive/fr_valid_reduced_1.csv')
df = dep_prune('fr', fr_test_data, 3)
df.to_csv('drive/MyDrive/fr_test_reduced_3.csv')
df = dep_prune('fr', fr_test_data, 2)
df.to_csv('drive/MyDrive/fr_test_reduced_2.csv')
df = dep_prune('fr', fr_test_data, 1)
df.to_csv('drive/MyDrive/fr_test_reduced_1.csv')

df = dep_prune('fr', fr_train_data, 3)
df.to_csv('drive/MyDrive/fr_train_reduced_3.csv')
df = dep_prune('fr', fr_train_data, 2)
df.to_csv('drive/MyDrive/fr_train_reduced_2.csv')
df = dep_prune('fr', fr_train_data, 1)
df.to_csv('drive/MyDrive/fr_train_reduced_1.csv')

