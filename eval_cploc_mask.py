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
from data_load_copy_mask import load_test_data, load_src_vocab, load_des_vocab
from train_copy_mask import Graph
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

hp.logdir = 'logdir_copy_mask'
result_dir = 'results_copy_mask'
extend_num = 4

def eval(): 
    # Load graph
    g = Graph(is_training=False, extend_num=extend_num)
    print("Graph loaded")
    
    # Load data
    X, Z, M, Sources, Targets = load_test_data(extend_num=extend_num)
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
                    z = Z[i*hp.batch_size: (i+1)*hp.batch_size]
                    m = M[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.y_maxlen), np.int32)
                    for j in range(hp.y_maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds, g.z: z, g.m: m})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    for source, target, z_, pred in zip(sources, targets, z, preds): # sentence-wise
                        got = []
                        target_words = np.array(target.split())
                        for idx in pred:
                            if idx in idx2des:
                                got.append(idx2des[idx])
                            else:
                                got.append(target_words[np.where(z_ == idx)[0]][0])
                        got = " ".join(got).split("</S>")[0].strip()
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
    
    
