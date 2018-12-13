# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

import sys
from hyperparams import Hyperparams as hp
from data_load_cploc_mask import get_batch_data, load_src_vocab, load_des_vocab
from modules import *
import os, codecs
from tqdm import tqdm

VERY_NEGATIVE = -1e6

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.xloc, self.yloc, self.m, self.num_batch = get_batch_data() # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.x_maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.y_maxlen))
                self.xloc = tf.placeholder(tf.int32, shape=(None, hp.x_maxlen))
                self.yloc = tf.placeholder(tf.int32, shape=(None, hp.y_maxlen))
                self.m = tf.placeholder(tf.int32, shape=(None, hp.x_maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

            # Load vocabulary    
            src2idx, idx2src = load_src_vocab()
            des2idx, idx2des = load_des_vocab()

            self.hidden_units = hp.hidden_units            

            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.x, 
                                     vocab_size=len(src2idx), 
                                     num_units=self.hidden_units, 
                                     scale=True,
                                     scope="enc_embed")
                self.enc_mask = tf.expand_dims(tf.cast(tf.equal(self.m, 1), tf.float32), 2)
                self.enc = tf.concat([self.enc, self.enc_mask], axis=2)
                self.hidden_units += 1
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=self.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.x_maxlen, 
                                      num_units=self.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                 
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=self.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*self.hidden_units, self.hidden_units])
            
            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs, 
                                     vocab_size=len(des2idx), 
                                     num_units=self.hidden_units,
                                     scale=True, 
                                     scope="dec_embed")
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.y_maxlen, 
                                      num_units=self.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.y_maxlen, 
                                      num_units=self.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                       keys=self.dec, 
                                                       num_units=self.hidden_units, 
                                                       num_heads=hp.num_heads, 
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True, 
                                                       scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                       keys=self.enc, 
                                                       num_units=self.hidden_units, 
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training, 
                                                       causality=False,
                                                       scope="vanilla_attention")

                        ## Feed Forward
                        with tf.variable_scope("num_blocks_fc_dec_{}".format(i)):
                            self.dec = feedforward(self.dec, num_units=[4*self.hidden_units, self.hidden_units])

            self.loc_enc = self.enc
            self.loc_logits = attention_matrix(queries=self.loc_enc,
                                            keys=self.dec, 
                                            num_units=self.hidden_units, 
                                            dropout_rate=hp.dropout_rate,
                                            is_training=is_training,
                                            causality=False, 
                                            scope="copy_matrix")

            xloc_vec = tf.one_hot(self.xloc, depth=hp.y_maxlen, dtype=tf.float32)
            yloc_vec = tf.one_hot(self.yloc, depth=hp.y_maxlen, dtype=tf.float32)
            loc_label = tf.matmul(yloc_vec, tf.transpose(xloc_vec, [0, 2, 1]))
            self.loc_label_history = tf.cumsum(loc_label, axis=1, exclusive=True)

            # Final linear projection
            self.loc_logits = tf.transpose(self.loc_logits, [0, 2, 1])

            self.loc_logits = tf.stack([self.loc_logits, self.loc_label_history], axis=3)
            self.loc_logits = tf.squeeze(tf.layers.dense(self.loc_logits, 1), axis=[3])

            x_masks = tf.tile(tf.expand_dims(tf.equal(self.x, 0), 1), [1, hp.y_maxlen, 1])
            #y_masks = tf.tile(tf.expand_dims(tf.equal(self.y, 0), -1), [1, 1, hp.x_maxlen])
            paddings = tf.ones_like(self.loc_logits)*(-1e6)
            self.loc_logits = tf.where(x_masks, paddings, self.loc_logits) # (N, T_q, T_k)
            #self.loc_logits = tf.where(y_masks, paddings, self.loc_logits) # (N, T_q, T_k)
            self.logits = tf.layers.dense(self.dec, len(des2idx))
            self.final_logits = tf.concat([self.logits, self.loc_logits], axis=2)
            #self.final_logits = tf.Print(self.final_logits, [self.final_logits[0][0][-3:]], message="final_logits_last")
            #self.final_logits = tf.Print(self.final_logits, [self.final_logits[0][0][:3]], message="final_logits_first")

            self.preds = tf.to_int32(tf.argmax(self.final_logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            

            if is_training:
                label = tf.one_hot(self.y, depth=len(des2idx), dtype=tf.float32)
                # A special case, when copy is open, we should not need unk label
                unk_pos = label[:,:,1]
                copy_pos = tf.sign(tf.reduce_sum(loc_label, axis=2))
                fix_pos = unk_pos * copy_pos
                #fix_pos = tf.Print(fix_pos, [tf.reduce_sum(unk_pos, axis=-1), tf.shape(unk_pos)], message="\nunk_pos", summarize=16)
                #fix_pos = tf.Print(fix_pos, [tf.reduce_sum(fix_pos, axis=-1), tf.shape(fix_pos)], message="\nfix_pos", summarize=16)
                fix_label = tf.expand_dims(label[:,:,1] - fix_pos, axis=2)
                label = tf.concat([label[:,:,:1], fix_label, label[:,:,2:]], axis=-1)

                self.final_label = tf.concat([label, loc_label], axis=2)
                #self.final_label = tf.Print(self.final_label, [self.final_label[0][0][-3:]], message="final_label")
                # Loss
                self.min_logit_loc = min_logit_loc = tf.argmax(self.final_logits + (-1e6)*(1.0-self.final_label), axis=-1)
                #min_logit_loc = tf.Print(min_logit_loc, [min_logit_loc[0]], message="min_logit_loc")
                self.min_label = tf.one_hot(min_logit_loc, depth=len(des2idx)+hp.x_maxlen, dtype=tf.float32)
                
                vocab_count = len(des2idx)+hp.x_maxlen-tf.reduce_sum(tf.cast(tf.equal(self.x, 0), dtype=tf.int32), axis=-1)
                #vocab_count = tf.Print(vocab_count, [vocab_count[0]], message="vocab_count")
                self.y_smoothed = label_smoothing_mask(self.min_label, vocab_count)
                #self.final_logits = tf.Print(self.final_logits, [self.final_logits[0][1][min_logit_loc[0][1]]], message="final_logits")
                #self.y_smoothed = tf.Print(self.y_smoothed, [self.y_smoothed[0][1][min_logit_loc[0][1]]], message="y_smoothed")
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_logits, labels=self.y_smoothed)
                #self.loss = tf.Print(self.loss, [self.final_label[0][1][min_logit_loc[0][1]]], message="final_label")
                #self.loss = tf.Print(self.loss, [self.loss[0][-3:]], message="loss_last")
                #self.loss = tf.Print(self.loss, [self.loss[0][:3]], message="loss_first")
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

if __name__ == '__main__':                
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth=True  

    # Load vocabulary    
    src2idx, idx2src = load_src_vocab()
    des2idx, idx2des = load_des_vocab()
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session(config=config) as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            print('Epoch ', epoch)
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, loss = sess.run([g.train_op, g.mean_loss])
                print('Loss = %s' % loss, end=' ')
                if loss > 100:
                    data = sess.run([g.y_smoothed, g.final_logits, g.final_label])
                    import pickle
                    pickle.dump(data, open('data.pkl', 'wb'))
                    exit()
                sys.stdout.flush()
                #input()
                
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_loss_%.3f_gs_%d' % (epoch, loss, gs))
    
    print("Done")    
    

