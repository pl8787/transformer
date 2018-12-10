from __future__ import print_function
import codecs
from hyperparams import Hyperparams as hp
import spacy
import jieba
import jieba.analyse
import pytextrank
import numpy as np
from tqdm import tqdm

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

def example():
    # Process whole documents
    text = (u"When Sebastian Thrun started working on self-driving cars at "
            u"Google in 2007, few people outside of the company took him "
            u"seriously. 'I can tell you very senior CEOs of major American "
            u"car companies would shake my hand and turn away because I wasn't "
            u"worth talking to,' said Thrun, now the co-founder and CEO of "
            u"online higher education startup Udacity, in an interview with "
            u"Recode earlier this week.")
    doc = nlp(text)
    
    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)

def search_entity(text):
    doc = nlp(text)
    pairs = []
    for entity in doc.ents:
        pairs.append([entity.text, entity.label_])
    return pairs

def search_keyword(text, context):
    words = jieba.analyse.extract_tags(text, topK=3, withWeight=False, allowPOS=())
    print(words)
    wset = set(context.split())
    words = [ w for w in words if w in wset ]
    #words = jieba.analyse.textrank(text, topK=1, withWeight=False, allowPOS=())
    return words

def search_textrank(text):
    text_dict = [{'id':'0', 'text':text}]
    for graf in pytextrank.parse_doc(text_dict):
        print(graf._asdict())

def search_idfword(text, context, idf_dict):
    wset = set(text.split()) & set(context.split())
    words = [ (w, idf_dict.get(w, [1000000, 1000000])[0]) for w in wset if idf_dict.get(w, [1000000, 1000000])[0] < 400000 ]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return words

def read_idf():
    fin = codecs.open('preprocessed/paragraph.vocab.tsv', 'r', 'utf-8')
    idf_dict = {}
    for rank, line in enumerate(fin):
        part = line.strip().split()
        idf_dict[part[0]] = (rank, part[1])
    return idf_dict

def tag_passage(q, p, idf_dict):
    words = search_idfword(q, p, idf_dict)
    p = np.array(p.split())
    p_mask = np.zeros(len(p), dtype=np.int32)
    for idx, (w, _) in enumerate(words):
        p_mask[p==w] = idx+1
    p_mask
    return p_mask

idf_dict = read_idf()

fin = codecs.open(hp.target_train, 'r', 'utf-8')
fin_c = codecs.open(hp.source_train, 'r', 'utf-8')
fout = codecs.open(hp.source_train_mask, 'w', 'utf-8')

for q in tqdm(fin):
    p = fin_c.readline().strip()
    q = q.strip()
    p_mask = tag_passage(q, p, idf_dict)
    fout.write(' '.join(list(map(str, p_mask))))
    fout.write('\n')

fout.close()

fin = codecs.open(hp.target_test, 'r', 'utf-8')
fin_c = codecs.open(hp.source_test, 'r', 'utf-8')
fout = codecs.open(hp.source_test_mask, 'w', 'utf-8')

for q in tqdm(fin):
    p = fin_c.readline().strip()
    q = q.strip()
    p_mask = tag_passage(q, p, idf_dict)
    fout.write(' '.join(list(map(str, p_mask))))
    fout.write('\n')

fout.close()
