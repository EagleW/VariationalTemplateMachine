from benchmark_reader import Benchmark
import os, json
from collections import Counter
from benchmark_reader import select_files
from nltk import word_tokenize
from tqdm import tqdm


def process_src(triples):
    src = ''
    for h, r, t in triples:
        h = ' '.join(word_tokenize(h))
        r = ' '.join(word_tokenize(r))
        t = ' '.join(word_tokenize(t))
        src += h + ' ' + r + ' ' + t + ' '
    return src[:-1]



def process_tgt_test(tgts):
    texts = []
    for text in tgts:
        tgt = word_tokenize(text.lex)
        texts.append(' '.join(tgt))
    return  texts

def convert_dataset_test(pair_src, pair_tgt, b):
    eids = []
    wf_src = open(pair_src, 'w')
    wf_tgt = open(pair_tgt, 'w')
    for entry in tqdm(b.entries):

        triples = entry.list_triples()
        if len(triples) == 0:
            continue
        cur_triples = []
        for triple in triples:
            h, r, t = triple.split(' | ')
            h = h.replace('"', '').replace('_', ' ')
            t = t.replace('"', '').replace('_', ' ')
            r = r.replace('"', '').replace('_', ' ')
            cur_triples.append((h,r,t))
        tgt = process_tgt_test(entry.lexs)
        if tgt == 0:
            continue
        eids.append(entry.id)
        src = process_src(cur_triples)
        wf_src.write(src + '\n')
        wf_tgt.write(json.dumps(tgt) + '\n')
    wf_tgt.close()
    wf_src.close()
    return eids

def convert_dataset_test_1(pair_src, pair_tgt, b, eids):
    wf_src = open(pair_src, 'w')
    wf_tgt = open(pair_tgt, 'w')
    for entry in tqdm(b.entries):
        if entry.id in eids:
            continue

        triples = entry.list_triples()
        if len(triples) == 0:
            continue
        cur_triples = []
        for triple in triples:
            h, r, t = triple.split(' | ')
            h = h.replace('"', '').replace('_', ' ')
            t = t.replace('"', '').replace('_', ' ')
            r = r.replace('"', '').replace('_', ' ')
            cur_triples.append((h,r,t))
        tgt = process_tgt_test(entry.lexs)
        src = process_src(cur_triples)
        wf_src.write(src + '\n')
        wf_tgt.write(json.dumps(tgt) + '\n')
    wf_tgt.close()
    wf_src.close()
    return eids


outdir = 'data/webnlg'

b = Benchmark()
files = [('webnlg_challenge_2017/test', 'testdata_unseen_with_lex.xml')]
b.fill_benchmark(files)

pair_valid_src = os.path.join(outdir, "unseen.source")
pair_valid_tgt = os.path.join(outdir, "unseen.target")
eids = convert_dataset_test(pair_valid_src, pair_valid_tgt, b)
a = Benchmark()
files =[('webnlg_challenge_2017/test', 'testdata_with_lex.xml')]
a.fill_benchmark(files)
pair_valid_src = os.path.join(outdir, "seen.source")
pair_valid_tgt = os.path.join(outdir, "seen.target")
eids = convert_dataset_test_1(pair_valid_src, pair_valid_tgt, a, eids)