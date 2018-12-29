# -*- coding: utf-8 -*-

class MarcoHyperparams():
    '''Hyperparameters'''
    def __init__(self, is_debug=False):
        # data
        self.root_dir = '/home/pangliang/set2list/data/marco_data/'
        if is_debug:
            self.root_dir += 'debug/'

        self.source_train = self.root_dir + 'train_v2.1_std.json.paragraph'
        self.target_train = self.root_dir + 'train_v2.1_std.json.question'
        self.source_train_mask = self.root_dir + 'train_v2.1_std.json.mask'
        self.ansloc_train = self.root_dir + 'train_v2.1_std.json.answer'

        self.source_test = self.root_dir + 'dev_v2.1_std.json.paragraph'
        self.target_test = self.root_dir + 'dev_v2.1_std.json.question'
        self.source_test_mask = self.root_dir + 'dev_v2.1_std.json.mask'
        self.ansloc_test = self.root_dir + 'dev_v2.1_std.json.answer'

        self.source_dev = self.root_dir + 'debug/' + 'train_v2.1_std.json.paragraph'
        self.target_dev = self.root_dir + 'debug/' + 'train_v2.1_std.json.question'
        self.source_dev_mask = self.root_dir + 'debug/' + 'train_v2.1_std.json.mask'
        self.ansloc_dev = self.root_dir + 'debug/' + 'train_v2.1_std.json.answer'

        self.src_vocab = self.root_dir + 'vocab/paragraph.marco.vocab.tsv'
        self.des_vocab = self.root_dir + 'vocab/question.marco.vocab.tsv'

        # training
        self.batch_size = 64 # alias = N
        self.lr = 0.0002 # learning rate. In paper, learning rate is adjusted to the global step.
        self.logdir = 'logdir_marco' # log directory

        # evaluation
        self.result_dir = 'result_marco'
        
        # model
        self.x_maxlen = 200 # Maximum number of words in a sentence. alias = T.
        self.y_maxlen = 20  # Maximum number of words in a sentence. alias = T.
                            # Feel free to increase this if you are ambitious.
        self.min_cnt = 60   # words whose occurred less than min_cnt are encoded as <UNK>.
        self.hidden_units = 511 # alias = C
        self.num_blocks = 4 # number of encoder/decoder blocks
        self.num_epochs = 20
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.sinusoid = False # If True, use sinusoid. If false, positional embedding.


class SquadHyperparams():
    '''Hyperparameters'''
    def __init__(self, is_debug=False):
        # data
        self.root_dir = '/home/pangliang/set2list/data/squad_data/'
        if is_debug:
            self.root_dir += 'debug/'

        self.source_train = self.root_dir + 'SQUAD_v2.0_train_std.json.paragraph'
        self.target_train = self.root_dir + 'SQUAD_v2.0_train_std.json.question'
        self.source_train_mask = self.root_dir + 'SQUAD_v2.0_train_std.json.mask'
        self.ansloc_train = self.root_dir + 'SQUAD_v2.0_train_std.json.answer'

        self.source_test = self.root_dir + 'SQUAD_v2.0_dev_std.json.paragraph'
        self.target_test = self.root_dir + 'SQUAD_v2.0_dev_std.json.question'
        self.source_test_mask = self.root_dir + 'SQUAD_v2.0_dev_std.json.mask'
        self.ansloc_test = self.root_dir + 'SQUAD_v2.0_dev_std.json.answer'

        self.source_dev = self.root_dir + 'debug/' + 'SQUAD_v2.0_train_std.json.paragraph'
        self.target_dev = self.root_dir + 'debug/' + 'SQUAD_v2.0_train_std.json.question'
        self.source_dev_mask = self.root_dir + 'debug/' + 'SQUAD_v2.0_train_std.json.mask'
        self.ansloc_dev = self.root_dir + 'debug/' + 'SQUAD_v2.0_train_std.json.answer'

        self.src_vocab = self.root_dir + 'vocab/paragraph.squad.vocab.tsv'
        self.des_vocab = self.root_dir + 'vocab/question.squad.vocab.tsv'

        # training
        self.batch_size = 64 # alias = N
        self.lr = 0.0002 # learning rate. In paper, learning rate is adjusted to the global step.
        self.logdir = 'logdir_squad' # log directory

        # evaluation
        self.result_dir = 'result_squad'
        
        # model
        self.x_maxlen = 100 # Maximum number of words in a sentence. alias = T.
        self.y_maxlen = 20  # Maximum number of words in a sentence. alias = T.
                            # Feel free to increase this if you are ambitious.
        self.min_cnt = 60   # words whose occurred less than min_cnt are encoded as <UNK>.
        self.hidden_units = 511 # alias = C
        self.num_blocks = 4 # number of encoder/decoder blocks
        self.num_epochs = 20
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
