# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import codecs
import regex
from tqdm import tqdm


class DataLoader():
    
    def __init__(self, hp):
        self.hp = hp

    def load_src_vocab(self):
        vocab = [line.split()[0] for line in codecs.open(self.hp.src_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=self.hp.min_cnt]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return word2idx, idx2word
    
    def load_des_vocab(self):
        vocab = [line.split()[0] for line in codecs.open(self.hp.des_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=self.hp.min_cnt]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return word2idx, idx2word
    
    def create_data(self, source_sents, target_sents, mask_sents, ans_locs): 
        src2idx, idx2src = self.load_src_vocab()
        des2idx, idx2des = self.load_des_vocab()
    
        # Index
        x_list, y_list, m_list, xloc_list, yloc_list, ans_start_list, ans_end_list, Sources, Targets = [], [], [], [], [], [], [], [], []
        line_id = 0
        for source_sent, target_sent, mask_sent, ans_loc in tqdm(zip(source_sents, target_sents, mask_sents, ans_locs)):
    
            source_sent = source_sent.split()
            target_sent = target_sent.split()
            mask_sent = mask_sent.split()
    
            if len(source_sent) >= self.hp.x_maxlen:
                source_sent = source_sent[:self.hp.x_maxlen-1]
            if len(target_sent) >= self.hp.y_maxlen:
                target_sent = target_sent[:self.hp.y_maxlen-1]
            if len(mask_sent) >= self.hp.x_maxlen:
                mask_sent = mask_sent[:self.hp.x_maxlen-1]
            if ans_loc[0] >= len(source_sent):
                ans_loc[0] = len(source_sent)-1
            if ans_loc[1] >= len(source_sent):
                ans_loc[1] = len(source_sent)-1
    
            assert ans_loc[0] < len(source_sent), "{}, {}, {}".format(line_id, ans_loc, len(source_sent))
            assert ans_loc[1] < len(source_sent), "{}, {}, {}".format(line_id, ans_loc, len(source_sent))
    
            xloc = np.zeros(self.hp.x_maxlen, dtype=np.int32) - 1
            yloc = np.zeros(self.hp.y_maxlen, dtype=np.int32) - 1
    
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
            ans_start_list.append(ans_loc[0])
            ans_end_list.append(ans_loc[1])
            Sources.append(" ".join(source_sent))
            Targets.append(" ".join(target_sent))
            
            line_id += 1
        
        # Pad      
        X = np.zeros([len(x_list), self.hp.x_maxlen], np.int32)
        Y = np.zeros([len(y_list), self.hp.y_maxlen], np.int32)
        M = np.zeros([len(m_list), self.hp.x_maxlen], np.int32)
        XLoc = np.array(xloc_list)
        YLoc = np.array(yloc_list)
        AnsStart = np.array(ans_start_list)
        AnsEnd = np.array(ans_end_list)
        for i, (x, y, m) in enumerate(zip(x_list, y_list, m_list)):
            X[i] = np.lib.pad(x, [0, self.hp.x_maxlen-len(x)], 'constant', constant_values=(0, 0))
            Y[i] = np.lib.pad(y, [0, self.hp.y_maxlen-len(y)], 'constant', constant_values=(0, 0))
            M[i] = np.lib.pad(m, [0, self.hp.x_maxlen-len(m)], 'constant', constant_values=(0, 0))
    
        return X, Y, XLoc, YLoc, M, AnsStart, AnsEnd, Sources, Targets
    
    def load_train_data(self):
        src_sents = [line for line in codecs.open(self.hp.source_train, 'r', 'utf-8').readlines()]
        des_sents = [line for line in codecs.open(self.hp.target_train, 'r', 'utf-8').readlines()]
        mask_sents = [line for line in codecs.open(self.hp.source_train_mask, 'r', 'utf-8').readlines()]
        ans_locs = [list(map(int, line.split())) for line in codecs.open(self.hp.ansloc_train, 'r', 'utf-8').readlines()]
        
        X, Y, XLoc, YLoc, M, AnsStart, AnsEnd, Sources, Targets = self.create_data(src_sents, des_sents, mask_sents, ans_locs)
        return X, Y, XLoc, YLoc, M, AnsStart, AnsEnd
        
    def load_test_data(self):
        src_sents = [line for line in codecs.open(self.hp.source_test, 'r', 'utf-8').readlines()]
        des_sents = [line for line in codecs.open(self.hp.target_test, 'r', 'utf-8').readlines()]
        mask_sents = [line for line in codecs.open(self.hp.source_test_mask, 'r', 'utf-8').readlines()]
        ans_locs = [list(map(int, line.split())) for line in codecs.open(self.hp.ansloc_test, 'r', 'utf-8').readlines()]
            
        X, Y, XLoc, YLoc, M, AnsStart, AnsEnd, Sources, Targets = self.create_data(src_sents, des_sents, mask_sents, ans_locs)
        return X, XLoc, M, AnsStart, AnsEnd, Sources, Targets # (1064, 150)
    
    def load_dev_data(self):
        src_sents = [line for line in codecs.open(self.hp.source_dev, 'r', 'utf-8').readlines()]
        des_sents = [line for line in codecs.open(self.hp.target_dev, 'r', 'utf-8').readlines()]
        mask_sents = [line for line in codecs.open(self.hp.source_dev_mask, 'r', 'utf-8').readlines()]
        ans_locs = [list(map(int, line.split())) for line in codecs.open(self.hp.ansloc_dev, 'r', 'utf-8').readlines()]
            
        X, Y, XLoc, YLoc, M, AnsStart, AnsEnd, Sources, Targets = self.create_data(src_sents, des_sents, mask_sents, ans_locs)
        return X, XLoc, M, AnsStart, AnsEnd, Sources, Targets # (1064, 150)
    
    def get_batch_data(self):
        # Load data
        X, Y, XLoc, YLoc, M, AnsStart, AnsEnd = self.load_train_data()
        
        # calc total batch count
        num_batch = len(X) // self.hp.batch_size
        print("Instance Num =", len(X))
        print("Batch Num =", num_batch)
        
        # Convert to tensor
        X = tf.convert_to_tensor(X, tf.int32)
        Y = tf.convert_to_tensor(Y, tf.int32)
        XLoc = tf.convert_to_tensor(XLoc, tf.int32)
        YLoc = tf.convert_to_tensor(YLoc, tf.int32)
        M = tf.convert_to_tensor(M, tf.int32)
        AnsStart = tf.convert_to_tensor(AnsStart, tf.int32)
        AnsEnd = tf.convert_to_tensor(AnsEnd, tf.int32)
        
        # Create Queues
        input_queues = tf.train.slice_input_producer([X, Y, XLoc, YLoc, M, AnsStart, AnsEnd])
                
        # create batch queues
        x, y, xloc, yloc, m, ans_start, ans_end = tf.train.shuffle_batch(input_queues,
                                    num_threads=8,
                                    batch_size=self.hp.batch_size, 
                                    capacity=self.hp.batch_size*64,   
                                    min_after_dequeue=self.hp.batch_size*32, 
                                    allow_smaller_final_batch=False)
        
        return x, y, xloc, yloc, m, ans_start, ans_end, num_batch # (N, T), (N, T), ()

