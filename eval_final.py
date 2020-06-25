import pickle
import collections
import sys
import json
from collections import Counter
sys.path.append('pycocoevalcap')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
#from pycocoevalcap.cider.cider import Cider

class Evaluate(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]#,        (Cider(), "CIDEr")

    def convert(self, data):
        if isinstance(data, basestring):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, get_scores=True, live=False, **kwargs):
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')

            # load caption data
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        # make dictionary
        hypo = {}
        ref = {}
        i = 0
        for vid, caption in cand.items():
            hypo[i] = [caption]
            ref[i] = temp_ref[vid]
            i += 1

        # compute scores
        final_scores = self.score(ref, hypo)
        # """
        # print out scores
        print ('Bleu_1:\t', final_scores['Bleu_1'])
        print ('Bleu_2:\t', final_scores['Bleu_2'])
        print ('Bleu_3:\t', final_scores['Bleu_3'])
        print ('Bleu_4:\t', final_scores['Bleu_4'])
        print ('METEOR:\t', final_scores['METEOR'])
        print ('ROUGE_L:', final_scores['ROUGE_L'])
        # print ('CIDEr:\t', final_scores['CIDEr'])
        # """

        if get_scores:
            return final_scores

def get_ref(path='data/webnlg/test_refs.txt'):
    i = 0
    refs = {}
    entities = {}
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.rstrip())
            refs[i] = data[0]
            entities[i] = data[1]
            i += 1
    return refs, entities

def get_cand(entities, path='result.txt'):
    i = 0
    j = 0
    cands = {}
    majority = Counter()
    with open(path, 'r') as f:
        for line in f:
            i += 1
            text = line.rstrip()
            if i % 6 == 0:
                text = majority.most_common(1)[0][0]
                majority = Counter()
                cands[j] = text
                j += 1
            else:
                majority[text] += 1
                for e in entities[j]:
                    if e in text:
                        majority[text] += 1
    return cands

def get_ref_1(path='data/Wiki/test_refs.txt'):
    i = 0
    j = 0
    refs = {}
    with open(path, 'r') as f:
        for line in f:
            if i%2 == 0:
                refs[j] = [line.rstrip()]
                j+= 1
            i += 1
    return refs

def get_cand_1(entities, path='result.txt'):
    i = 0
    j = 0
    cands = {}
    majority = Counter()
    with open(path, 'r') as f:
        for line in f:
            i += 1
            text = line.rstrip()
            if i % 6 == 0:
                text = majority.most_common(1)[0][0]
                majority = Counter()
                cands[j] = text
                j += 1
            else:
                majority[text] += 1
                for e in entities[j]:
                    if e in text:
                        majority[text] += 1
    return cands


def get_entities(path='data/Wiki/src_test.txt'):
    entities = {}
    i = 0
    with open(path, 'r') as f:
        for line in f:
            tmp = []
            text = line.rstrip()
            for ee in text.split('\t'):
                _, _, e = ee.partition(':')
                if e != '<none>':
                    tmp.append(e)
            entities[i] = tmp
            i += 1
    return entities




if __name__ == '__main__':
    # cand = {'generated_description1': 'how are you', 'generated_description2': 'Hello how are you'}
    # ref = {'generated_description1': ['what are you', 'where are you'],
    #        'generated_description2': ['Hello how are you', 'Hello how is your day']}
    ref = get_ref_1()
    entities = get_entities()
    cand = get_cand_1(entities)
    # ref, entities = get_ref()
    # cand = get_cand(entities)
    x = Evaluate()
    x.evaluate(live=True, cand=cand, ref=ref)
