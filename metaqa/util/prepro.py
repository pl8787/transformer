# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter


class Vocab():

    def __init__(self, hp):
        self.hp = hp

    def make_vocab(self, fpath, fname):
        '''Constructs vocabulary.
        
        Args:
          fpath: A string. Input file path.
          fname: A string. Output file name.
        
        Writes vocabulary line by line to `preprocessed/fname`
        '''  
        text = codecs.open(fpath, 'r', 'utf-8').read()
        words = text.split()
        word2cnt = Counter(words)
        with codecs.open(fname, 'w', 'utf-8') as fout:
            fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
            for word, cnt in word2cnt.most_common(len(word2cnt)):
                fout.write(u"{}\t{}\n".format(word, cnt))
    
    def run(self):
        self.make_vocab(self.hp.source_train, self.hp.src_vocab)
        self.make_vocab(self.hp.target_train, self.hp.des_vocab)
        print("Done")
