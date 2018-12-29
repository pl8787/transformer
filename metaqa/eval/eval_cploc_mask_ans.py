# -*- coding: utf-8 -*-
import codecs
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from metaqa.loader.data_load_cploc_mask_ans import DataLoader
from metaqa.train.train_cploc_mask_ans import Graph
from nltk.translate.bleu_score import corpus_bleu

class Evaluator():

    def __init__(self, hp, dl):
        self.hp = hp
        self.dl = dl

    def remove_dup(self, x):
        if len(x) == 1:
            return x
        y = [x[0]]
        for w in x[1:]:
            if w == y[-1]:
                continue
            else:
                y.append(w)
        return y
    
    def eval(self, stage='test', checkpoint_file=None, is_dedup=False, clue_level=1): 
        # Load graph
        g = Graph(hp=self.hp, dl=self.dl, is_training=False, clue_level=clue_level)
        print("Graph loaded")
        
        # Load data
        if stage == 'test':
            X, XLoc, M, AnsStart, AnsEnd, Sources, Targets = self.dl.load_test_data()
        else:   
            X, XLoc, M, AnsStart, AnsEnd, Sources, Targets = self.dl.load_dev_data()
    
        src2idx, idx2src = self.dl.load_src_vocab()
        des2idx, idx2des = self.dl.load_des_vocab()
         
        # X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
    
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth=True  
        config.allow_soft_placement=True
    
        num_gen = 0
        num_copy = 0     
        num_unk_copy = 0
        max_batch = 10
    
        # Start session         
        with g.graph.as_default():    
            sv = tf.train.Supervisor()
            with sv.managed_session(config=config) as sess:
                if not checkpoint_file:
                    checkpoint_file = tf.train.latest_checkpoint(self.hp.logdir)
    
                ## Restore parameters
                sv.saver.restore(sess, checkpoint_file)
                print("Restored! {}".format(checkpoint_file))
                  
                ## Get model name
                #mname = open(self.hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
                mname = checkpoint_file.split('/')[1]
                 
                ## Inference
                if not os.path.exists(self.hp.result_dir): os.mkdir(self.hp.result_dir)
                with codecs.open(self.hp.result_dir + "/" + mname + '.level{}'.format(clue_level) + '.' + stage, "w", "utf-8") as fout:
                    list_of_refs, hypotheses = [], []
                    for i in tqdm(range(min(max_batch, len(X) // self.hp.batch_size))):
                         
                        ### Get mini-batches
                        x = X[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                        xloc = XLoc[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                        m = M[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                        ans_start = AnsStart[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                        ans_end = AnsEnd[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                        sources = Sources[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                        targets = Targets[i*self.hp.batch_size: (i+1)*self.hp.batch_size]
                         
                        ### Autoregressive inference
                        preds = np.zeros((self.hp.batch_size, self.hp.y_maxlen), np.int32)
                        preds_unk = np.zeros((self.hp.batch_size, self.hp.y_maxlen), np.int32)
                        preds_xloc = np.zeros((self.hp.batch_size, self.hp.x_maxlen), np.int32) - 1
                        preds_yloc = np.zeros((self.hp.batch_size, self.hp.y_maxlen), np.int32) - 1
    
                        for j in range(self.hp.y_maxlen):
                            _preds, ans_start_pred, ans_end_pred, loc_logits = \
                                sess.run([g.preds, g.ans_start_preds, g.ans_end_preds, g.loc_logits], 
                                    {g.x: x, g.y: preds_unk, g.m: m, g.xloc: preds_xloc, g.yloc: preds_yloc})
                            preds[:, j] = _preds[:, j]
                            
                            preds_unk[:, j] = _preds[:, j]
                            preds_unk[preds_unk>=len(idx2des)] = 1
    
                            for k in range(self.hp.batch_size):
                                xloc = np.zeros(self.hp.x_maxlen, dtype=np.int32) - 1
                                yloc = np.zeros(self.hp.y_maxlen, dtype=np.int32) - 1
    
                                source_words = sources[k].split()
                                target_words = []
                                for idx in preds[k]:
                                    if idx in idx2des:
                                        target_words.append(idx2des[idx])
                                    elif idx - len(idx2des) == len(source_words):
                                        target_words.append('</S>')
                                    else:
                                        cp_word_idx = idx - len(idx2des)
                                        cp_word = source_words[cp_word_idx]
                                        target_words.append(cp_word)
                                source_sent_np = np.array(source_words)
                                target_sent_np = np.array(target_words)
                                source_wset = set(source_words)
                                target_wset = set(target_words)
                                for loc_id, w in enumerate(target_wset & source_wset):
                                    xloc[np.where(source_sent_np==w)] = loc_id
                                    yloc[np.where(target_sent_np==w)] = loc_id
                                preds_xloc[k] = xloc 
                                preds_yloc[k] = yloc
                            #print(loc_logits.shape)
                            #print(loc_logits[0][j][:20])
                            #input()
    
                        ### Write to file
                        for source, target, m_, pred, ans_s_p, ans_e_p, ans_s, ans_e in zip(sources, targets, m, preds, ans_start_pred, ans_end_pred, ans_start, ans_end): # sentence-wise
                            got_display = []
                            got = []
                            ans = []
                            source_words = np.array(source.split())
                            for idx in pred:
                                if idx in idx2des:
                                    num_gen += 1
                                    got.append(idx2des[idx])
                                    got_display.append(idx2des[idx]+'[{}]'.format(idx))
                                else:
                                    num_copy += 1
                                    cp_word_idx = idx - len(idx2des)
                                    cp_word = source_words[cp_word_idx]
                                    got.append(cp_word)
                                    got_display.append(cp_word+'[{},{}]'.format(cp_word_idx, m_[cp_word_idx]))
                                    if cp_word not in des2idx:
                                        num_unk_copy += 1
                            
                            if is_dedup:
                                got = self.remove_dup(got)
                                got_display = self.remove_dup(got_display)
    
                            got = " ".join(got).split("</S>")[0].strip()
                            got_display = " ".join(got_display).split("</S>")[0].strip()
                            ans = " ".join(source_words[ans_s_p:ans_e_p+1])

                            fout.write("- source: " + source +"\n")
                            fout.write("- expected: " + target + "\n")
                            fout.write("- got: " + got + "\n")
                            fout.write("- analyse: " + got_display + "\n")
                            fout.write("- ans pos: gt({}, {}), pred({}, {})".format(ans_s, ans_e, ans_s_p, ans_e_p) + "\n")
                            fout.write("- ans: " + ans + "\n\n")
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
                    fout.write("Generate / Copy / UNK Copy = {} / {} / {}".format(num_gen, num_copy, num_unk_copy))
                                              
def run(hp):
    dl = DataLoader(hp)
    e = Evaluator(hp, dl)

    all_models = open(hp.logdir + '/checkpoint', 'r').readlines()
    all_models = [hp.logdir + '/' + all_models[i].split('"')[1] for i in range(len(all_models))]

    for f in all_models[1:]:
        for clue_level in [1, 2, 3, 5]:
            e.eval('test', f, True, clue_level)
            e.eval('dev', f, True, clue_level)
    
    print("Done")
