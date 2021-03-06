# -*- coding: utf-8 -*-
"""intent_identify_DL_bert _kochat_normal_news.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10dWcfb4iYmK-8BEZEWiFCdqiWNcFWPKb
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import unicodedata
import six
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_radam import RAdam
import os
import pandas as pd
import numpy as np
import random
import json
import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


data = pd.read_csv('./Intent_cl/intent_dataset/category_data')
data_intent = pd.DataFrame(columns={'sentence' , 'intent' })
data_intent['sentence'] = data['question']
data_intent['intent']= data['label']
data_intent.reset_index(drop=True, inplace = True)

"""ETIR korbert model API 신청 후, 제공받은 모델 중  bert_eojeol_tensorflow 디렉토리를 
 /intent_data/Bert_model 경로에 넣어주시면 됩니다.  """

pretrained_path = './Intent_cl/Bert_model/004_bert_eojeol_tensorflow'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'model.ckpt-56000')
vocab_path = os.path.join(pretrained_path, 'vocab.korean.rawtext.list')
layer_num = 12
model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training = True,
    trainable = True
)

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.io.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break

      ### joonho.lim @ 2019-03-15
      if token.find('n_iters=') == 0 or token.find('max_length=') == 0 :
        continue
      token = token.split('\t')[0]

      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab

def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output

def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)

def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)

def convert_ids_to_tokens2(inv_vocab, ids):
  output = []
  for idss in ids:
    for key, valus in inv_vocab.items():
      if valus == idss:
        output.append(key)
  return output

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      ### joonho.lim @ 2019-03-15
      token += '_'
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    output_tokens.insert(0, '[CLS]')
    output_tokens.append('[SEP]')
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def _is_punctuation(char):

  return char == ' '

with open('./Bert_model/vocab.json', 'r') as read_file:
    vocab = json.loads(read_file.read())


"""wordpiecetokenizer를 사용합니다."""
a = WordpieceTokenizer(vocab)

token_sentence=[]
for i in data_intent['sentence']:
  token_sentence.append(a.tokenize(i))

token_ids = []
for i in token_sentence:
  token_ids.append(convert_by_vocab(vocab, i))


"""## input으로 들어갈 데이터 텍스트 토큰화, 인덱싱 작업입니다."""

import numpy as np
classes = data_intent.intent.unique().tolist()
token_ids = []

"""x_input 토큰 생성 코드입니다."""
for i in token_sentence:
  token_ids.append(convert_by_vocab(vocab, i))
max_len = max([len(i) for i in token_ids])
for i in range(len(token_sentence)):
  num = max_len - len(token_sentence[i])
  if num == 0:
    pass
  else: 
    for x in range(num):
          token_ids[i].append(0)
token_ids = np.array(token_ids)

"""intent target data 생성입니다."""
y = []
for intent in data_intent['intent']:
  y.append(classes.index(intent))
intent_val = np.array(y)

"""## 모델 생성함수입니다. 기존의 pretrained Korbert 모델을 호출하여 bert레이어를 만들고, input과 output을 
저희 팀에서 추구하는 category_cl, Intent_cl task를 수행할 수 있도록 수정해나가는 방식입니다.
기초 코드는 https://www.kdnuggets.com/2020/02/intent-recognition-bert-keras-tensorflow.html 참고한 소스입니다.
"""
def create_model(max_seq_len, bert_ckpt_file):
  with tf.io.gfile.GFile(config_path, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)
        
  return model

"""ETRI korbert 기반으로 딥러닝 모델생성 부분입니다."""
model = create_model(max_len, checkpoint_path)

"""optimizer, loss_fn, metrics 및 compile 속성 선언 부분입니다. """
model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

"""## fitting -> fine tuning 과정입니다. Colab GPU 런타임 기준 약 50분정도 소요되었습니다.
"""
history = model.fit(
  x= token_ids,
  y= intent_val,
  validation_split=0.1,
  batch_size=10,
  shuffle=True,
  epochs=10,
)

"""## finetuning 된 모델 저장입니다. finetuning 및 fitting이 완료된 모델이기 떄문에, 훈련이 완료되면 호출만 하여 구분합니다."""
model.save('./Intent_cl/Bert_model/kobert_model_category.h5')


"""훈련된 모델에 sentence를 넣어봄으로써 성능을확인할 수 있는 부분입니다."""
from tqdm import tqdm
def prediction_text_frame(text_frame):
  pre = 0
  for i in tqdm(text_frame['sentence']):
    padding_text = WordpieceTokenizer(vocab)
    a= padding_text.tokenize(i)
    token_ids = convert_by_vocab(vocab, a)
    for x in range(max_len - len(token_ids)):
      token_ids.append(0)
    token_ids = np.array(token_ids)
    token_ids = token_ids.reshape(1,max_len, order= 'F')
    predictions = model.predict(token_ids).argmax(axis=-1)
    if pre != predictions:
          print('\ntext :', i,'\nintent :', classes[int(predictions)])
          pre = predictions



"""## 기존 데이터 셋에서 랜덤으로 100개의 샘플을 구해와서 카테고리 구분을 해보았습니다."""
prediction_text_frame(data_intent.sample(n = 100))

