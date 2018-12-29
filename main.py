import sys
import os

from metaqa.util.hyperparams \
    import SquadHyperparams \
    as Hyperparams
from metaqa.util.prepro \
    import Vocab
from metaqa.util.core_words \
    import CoreWords
from metaqa.train \
    import train_cploc_mask_ans \
    as train
from metaqa.eval \
    import eval_cploc_mask_ans \
    as evaluation

hp = Hyperparams(is_debug=True)

'''
vocab = Vocab(hp)
vocab.run()

core_words = CoreWords(hp)
core_words.run()
'''

train.run(hp)

evaluation.run(hp)
