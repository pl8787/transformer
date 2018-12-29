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

def create_data(source_sents, target_sents, mask_sents, extend_num=-1): 
    src2idx, idx2src = load_src_vocab()
    des2idx, idx2des = load_des_vocab()

    max_extend_num = 0    

    # Index
    x_list, y_list, z_list, m_list, Sources, Targets = [], [], [], [], [], []
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

        oov2idx = {}
        source_wset = set(source_sent)
        for w in target_sent:
            if extend_num > 0 and len(oov2idx) == extend_num:
                break
            if w not in des2idx and w in source_wset and w not in oov2idx:
                oov2idx[w] = len(oov2idx) + len(des2idx)

        max_extend_num = max(max_extend_num, len(oov2idx))

        oov2idx.update(des2idx)
        x = [src2idx.get(word, 1) for word in (source_sent + [u"</S>"])] # 1: OOV, </S>: End of Text
        y = [des2idx.get(word, 1) for word in (target_sent + [u"</S>"])] 
        z = [oov2idx.get(word, 1) for word in (target_sent + [u"</S>"])]
        m = [int(c) for c in mask_sent]

        x_list.append(np.array(x))
        y_list.append(np.array(y))
        z_list.append(np.array(z))
        m_list.append(np.array(m))
        Sources.append(" ".join(source_sent))
        Targets.append(" ".join(target_sent))
    
    # Pad      
    X = np.zeros([len(x_list), hp.x_maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.y_maxlen], np.int32)
    Z = np.zeros([len(y_list), hp.y_maxlen], np.int32)
    M = np.zeros([len(m_list), hp.x_maxlen], np.int32)
    for i, (x, y, z, m) in enumerate(zip(x_list, y_list, z_list, m_list)):
        X[i] = np.lib.pad(x, [0, hp.x_maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.y_maxlen-len(y)], 'constant', constant_values=(0, 0))
        Z[i] = np.lib.pad(z, [0, hp.y_maxlen-len(y)], 'constant', constant_values=(0, 0))
        M[i] = np.lib.pad(m, [0, hp.x_maxlen-len(m)], 'constant', constant_values=(0, 0))

    print("[Read Data] max_extend_num =", max_extend_num)    

    return X, Y, Z, M, Sources, Targets, max_extend_num

def load_train_data():
    src_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n")]
    des_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n")]
    mask_sents = [line for line in codecs.open(hp.source_train_mask, 'r', 'utf-8').read().split("\n")]
    
    X, Y, Z, M, Sources, Targets, max_extend_num = create_data(src_sents, des_sents, mask_sents)
    return X, Y, Z, M, max_extend_num
    
def load_test_data(extend_num=-1):
    src_sents = [line for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    des_sents = [line for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
    mask_sents = [line for line in codecs.open(hp.source_test_mask, 'r', 'utf-8').read().split("\n")]
        
    X, Y, Z, M, Sources, Targets, max_extend_num = create_data(src_sents, des_sents, mask_sents, extend_num=extend_num)
    return X, Z, M, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y, Z, M, max_extend_num = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    print("Instance Num =", len(X))
    print("Batch Num =", num_batch)
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    Z = tf.convert_to_tensor(Z, tf.int32)
    M = tf.convert_to_tensor(M, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y, Z, M])
            
    # create batch queues
    x, y, z, m = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, z, m, max_extend_num, num_batch # (N, T), (N, T), ()

