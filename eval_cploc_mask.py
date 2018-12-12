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
from data_load_cploc_mask import load_test_data, load_src_vocab, load_des_vocab
from train_cploc_mask import Graph
#from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

hp.logdir = 'logdir_cploc_mask'
result_dir = 'results_cploc_mask'

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, XLoc, M, Sources, Targets = load_test_data()
    src2idx, idx2src = load_src_vocab()
    des2idx, idx2des = load_des_vocab()
     
    # X, Sources, Targets = X[:33], Sources[:33], Targets[:33]

    num_gen = 0
    num_copy = 0     
    num_unk_copy = 0
    num_batch = 0
    max_batch = 10

    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            with codecs.open(result_dir + "/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in tqdm(range(len(X) // hp.batch_size)):
                     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    xloc = XLoc[i*hp.batch_size: (i+1)*hp.batch_size]
                    m = M[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.y_maxlen), np.int32)
                    preds_unk = np.zeros((hp.batch_size, hp.y_maxlen), np.int32)
                    for j in range(hp.y_maxlen):
                        _preds, loc_logits = sess.run([g.preds, g.loc_logits], {g.x: x, g.y: preds_unk, g.m: m})
                        preds[:, j] = _preds[:, j]
                        
                        preds_unk[:, j] = _preds[:, j]
                        preds_unk[preds_unk>=len(idx2des)] = 1

                        #print(loc_logits.shape)
                        #print(loc_logits[0][j][:20])
                        #input()
                     
                    ### Write to file
                    for source, target, m_, pred in zip(sources, targets, m, preds): # sentence-wise
                        got = []
                        source_words = np.array(source.split())
                        for idx in pred:
                            if idx in idx2des:
                                num_gen += 1
                                got.append(idx2des[idx])
                            else:
                                num_copy += 1
                                cp_word_idx = idx - len(idx2des)
                                cp_word = source_words[cp_word_idx]
                                got.append(cp_word+'[{},{}]'.format(cp_word_idx, m_[cp_word_idx]))
                                if cp_word not in des2idx:
                                    num_unk_copy += 1
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

                    num_batch += 1
                    if num_batch > max_batch:
                        break
                ## Calculate bleu score
                #score = corpus_bleu(list_of_refs, hypotheses)
                #fout.write("Bleu Score = " + str(100*score) + "\n")
                #score = corpus_bleu(list_of_refs, hypotheses, weights=(1, 0, 0, 0))
                #fout.write("Bleu@1 Score = " + str(100*score) + "\n")
                #score = corpus_bleu(list_of_refs, hypotheses, weights=(0, 1, 0, 0))
                #fout.write("Bleu@2 Score = " + str(100*score) + "\n")
                #score = corpus_bleu(list_of_refs, hypotheses, weights=(0, 0, 1, 0))
                #fout.write("Bleu@3 Score = " + str(100*score) + "\n")
                #score = corpus_bleu(list_of_refs, hypotheses, weights=(0, 0, 0, 1))
                #fout.write("Bleu@4 Score = " + str(100*score) + "\n")
                fout.write("Generate / Copy / UNK Copy = {} / {} / {}".format(num_gen, num_copy, num_unk_copy))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
