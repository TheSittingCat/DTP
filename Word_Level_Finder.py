#@title Word Levels 
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