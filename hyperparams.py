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
    
    root_dir = '/home/pangliang/set2list/data/marco_data/'
    #root_dir = '/home/pangliang/set2list/data/question_generation/debug/'

    source_train = root_dir + 'train_v2.1_std.json.paragraph'
    target_train = root_dir + 'train_v2.1_std.json.question'
    source_train_mask = root_dir + 'train_v2.1_std.json.mask'
    ansloc_train = root_dir + 'train_v2.1_std.json.answer'

    source_test = root_dir + 'dev_v2.1_std.json.paragraph'
    target_test = root_dir + 'dev_v2.1_std.json.question'
    source_test_mask = root_dir + 'dev_v2.1_std.json.mask'
    ansloc_test = root_dir + 'dev_v2.1_std.json.answer'

    source_dev = root_dir + 'debug/' + 'train_v2.1_std.json.paragraph'
    target_dev = root_dir + 'debug/' + 'train_v2.1_std.json.question'
    source_dev_mask = root_dir + 'debug/' + 'train_v2.1_std.json.mask'
    ansloc_dev = root_dir + 'debug/' + 'train_v2.1_std.json.answer'
    #source_test = root_dir + 'train_v2.1_th_std.json.paragraph'
    #target_test = root_dir + 'train_v2.1_th_std.json.question'

    # training
    batch_size = 64 # alias = N
    lr = 0.0002 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir_marco' # log directory
    
    # model
    x_maxlen = 200 # Maximum number of words in a sentence. alias = T.
    y_maxlen = 20 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 60 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 511 # alias = C
    num_blocks = 4 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    
