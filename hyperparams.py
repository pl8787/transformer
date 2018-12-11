# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    #source_train = 'corpora/train.tags.de-en.de'
    #target_train = 'corpora/train.tags.de-en.en'
    #source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    #target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    
    root_dir = '/home/yuxiaoming/qgeneration/data/question_generation/'
    source_train = root_dir + 'train_v2.1_qg_std.json.paragraph'
    target_train = root_dir + 'train_v2.1_qg_std.json.question'
    source_train_mask = root_dir + 'train_v2.1_qg_std.json.mask'
    source_test = root_dir + 'dev_v2.1_std.json.paragraph'
    target_test = root_dir + 'dev_v2.1_std.json.question'
    source_test_mask = root_dir + 'dev_v2.1_std.json.mask'
    #source_test = root_dir + 'train_v2.1_th_std.json.paragraph'
    #target_test = root_dir + 'train_v2.1_th_std.json.question'

    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir_cploc_mask_v1' # log directory
    
    # model
    x_maxlen = 200 # Maximum number of words in a sentence. alias = T.
    y_maxlen = 20 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 511 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    
