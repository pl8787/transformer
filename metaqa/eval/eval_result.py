import sys
from bleu_metric.bleu import Bleu

def cal_bleu(in_path):
    data_file = open(in_path, 'r', encoding='utf-8')
    prediction = dict()
    ref = dict()
    i = 0
    j = 0
    for line in data_file:
        if '- expected: ' in line:
            gold = line.strip('- expected: ')
            ref[i] = [gold]
            i += 1
        if '- got: ' in line:
            pred = line.strip('- got: ')
            prediction[j] = [pred]
            j += 1
    print(len(prediction))
    print(len(ref))
    bleu_scores, _ = \
        Bleu(4).compute_score(ref, prediction)
    print(bleu_scores)

cal_bleu(sys.argv[1])
