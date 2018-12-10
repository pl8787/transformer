# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load_mask import load_test_data, load_src_vocab, load_des_vocab
from train_mask import Graph
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

hp.logdir = 'logdir_mask'
result_dir = 'results_mask'

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, M, Sources, Targets = load_test_data()
    src2idx, idx2src = load_src_vocab()
    des2idx, idx2des = load_des_vocab()
     
    # X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint_best', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            with codecs.open(result_dir + "/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in tqdm(range(len(X) // hp.batch_size)):
                     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    m = M[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.y_maxlen), np.int32)
                    for j in range(hp.y_maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds, g.m: m})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2des[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                          
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score) + "\n")
                score = corpus_bleu(list_of_refs, hypotheses, weights=(1, 0, 0, 0))
                fout.write("Bleu@1 Score = " + str(100*score) + "\n")
                score = corpus_bleu(list_of_refs, hypotheses, weights=(0, 1, 0, 0))
                fout.write("Bleu@2 Score = " + str(100*score) + "\n")
                score = corpus_bleu(list_of_refs, hypotheses, weights=(0, 0, 1, 0))
                fout.write("Bleu@3 Score = " + str(100*score) + "\n")
                score = corpus_bleu(list_of_refs, hypotheses, weights=(0, 0, 0, 1))
                fout.write("Bleu@4 Score = " + str(100*score) + "\n")
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
