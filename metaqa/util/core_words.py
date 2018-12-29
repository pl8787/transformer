# -*- coding: utf-8 -*-

import codecs
import numpy as np
from tqdm import tqdm

class CoreWords():

    def __init__(self, hp):
        self.hp = hp

    def read_idf(self):
        fin = codecs.open('preprocessed/paragraph.vocab.tsv', 'r', 'utf-8')
        idf_dict = {}
        for rank, line in enumerate(fin):
            part = line.strip().split()
            idf_dict[part[0]] = (rank, part[1])
        return idf_dict
    
    def search_idfword(self, text, context, idf_dict):
        wset = set(text.split()) & set(context.split())
        words = [ (w, idf_dict.get(w, [1000000, 1000000])[0]) for w in wset if idf_dict.get(w, [1000000, 1000000])[0] < 400000 ]
        words = sorted(words, key=lambda x: x[1], reverse=True)
        return words
    
    def tag_passage(self, q, p, idf_dict):
        words = self.search_idfword(q, p, idf_dict)
        p = np.array(p.split())
        p_mask = np.zeros(len(p), dtype=np.int32)
        for idx, (w, _) in enumerate(words):
            p_mask[p==w] = idx+1
        p_mask
        return p_mask
    
    def run(self):
        idf_dict = self.read_idf()
        
        fin = codecs.open(self.hp.target_train, 'r', 'utf-8')
        fin_c = codecs.open(self.hp.source_train, 'r', 'utf-8')
        fout = codecs.open(self.hp.source_train_mask, 'w', 'utf-8')
        
        for q in tqdm(fin):
            p = fin_c.readline().strip()
            q = q.strip()
            p_mask = self.tag_passage(q, p, idf_dict)
            fout.write(' '.join(list(map(str, p_mask))))
            fout.write('\n')
        
        fout.close()
        
        fin = codecs.open(self.hp.target_test, 'r', 'utf-8')
        fin_c = codecs.open(self.hp.source_test, 'r', 'utf-8')
        fout = codecs.open(self.hp.source_test_mask, 'w', 'utf-8')
        
        for q in tqdm(fin):
            p = fin_c.readline().strip()
            q = q.strip()
            p_mask = self.tag_passage(q, p, idf_dict)
            fout.write(' '.join(list(map(str, p_mask))))
            fout.write('\n')
        
        fout.close()
