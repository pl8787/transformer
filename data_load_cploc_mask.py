# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
from tqdm import tqdm

def load_src_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/paragraph.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_des_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/question.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents, mask_sents): 
    src2idx, idx2src = load_src_vocab()
    des2idx, idx2des = load_des_vocab()

    # Index
    x_list, y_list, m_list, xloc_list, yloc_list, Sources, Targets = [], [], [], [], [], [], []
    for source_sent, target_sent, mask_sent in tqdm(zip(source_sents, target_sents, mask_sents)):

        source_sent = source_sent.split()
        target_sent = target_sent.split()
        mask_sent = mask_sent.split()

        if len(source_sent) >= hp.x_maxlen:
            source_sent = source_sent[:hp.x_maxlen-1]
        if len(target_sent) >= hp.y_maxlen:
            target_sent = target_sent[:hp.y_maxlen-1]
        if len(mask_sent) >= hp.x_maxlen:
            mask_sent = mask_sent[:hp.x_maxlen-1]

        xloc = np.zeros(hp.x_maxlen, dtype=np.int32) - 1
        yloc = np.zeros(hp.y_maxlen, dtype=np.int32) - 1

        source_sent_np = np.array(source_sent)
        target_sent_np = np.array(target_sent)
        source_wset = set(source_sent)
        target_wset = set(target_sent)
        for loc_id, w in enumerate(target_wset & source_wset):
            xloc[np.where(source_sent_np==w)] = loc_id
            yloc[np.where(target_sent_np==w)] = loc_id
        xloc_list.append(xloc) 
        yloc_list.append(yloc)

        x = [src2idx.get(word, 1) for word in (source_sent + [u"</S>"])] # 1: OOV, </S>: End of Text
        y = [des2idx.get(word, 1) for word in (target_sent + [u"</S>"])] 
        m = [int(c) for c in mask_sent]

        x_list.append(np.array(x))
        y_list.append(np.array(y))
        m_list.append(np.array(m))
        Sources.append(" ".join(source_sent))
        Targets.append(" ".join(target_sent))
    
    # Pad      
    X = np.zeros([len(x_list), hp.x_maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.y_maxlen], np.int32)
    M = np.zeros([len(m_list), hp.x_maxlen], np.int32)
    XLoc = np.array(xloc_list)
    YLoc = np.array(yloc_list)
    for i, (x, y, m) in enumerate(zip(x_list, y_list, m_list)):
        X[i] = np.lib.pad(x, [0, hp.x_maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.y_maxlen-len(y)], 'constant', constant_values=(0, 0))
        M[i] = np.lib.pad(m, [0, hp.x_maxlen-len(m)], 'constant', constant_values=(0, 0))

    return X, Y, XLoc, YLoc, M, Sources, Targets

def load_train_data():
    src_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n")]
    des_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n")]
    mask_sents = [line for line in codecs.open(hp.source_train_mask, 'r', 'utf-8').read().split("\n")]
    
    X, Y, XLoc, YLoc, M, Sources, Targets = create_data(src_sents, des_sents, mask_sents)
    return X, Y, XLoc, YLoc, M
    
def load_test_data():
    src_sents = [line for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    des_sents = [line for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
    mask_sents = [line for line in codecs.open(hp.source_test_mask, 'r', 'utf-8').read().split("\n")]
        
    X, Y, XLoc, YLoc, M, Sources, Targets = create_data(src_sents, des_sents, mask_sents)
    return X, XLoc, M, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y, XLoc, YLoc, M = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    print("Instance Num =", len(X))
    print("Batch Num =", num_batch)
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    XLoc = tf.convert_to_tensor(XLoc, tf.int32)
    YLoc = tf.convert_to_tensor(YLoc, tf.int32)
    M = tf.convert_to_tensor(M, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y, XLoc, YLoc, M])
            
    # create batch queues
    x, y, xloc, yloc, m = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, xloc, yloc, m, num_batch # (N, T), (N, T), ()

