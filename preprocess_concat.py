from benchmark_reader import Benchmark
import os, json
from collections import Counter
from benchmark_reader import select_files
import spacy
from nltk import word_tokenize
from tqdm import tqdm


# nlp = spacy.load("en_core_web_sm")
# def word_tokenize(txt):
#     words = []
#     doc = nlp(txt)
#     for token in doc:
#         words.append(token.text)
#     return words


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


def convert_dataset(pair_src, pair_tgt, b):
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
        if len(tgt) == 0:
            continue
        src = process_src(cur_triples)
        for tg in tgt:
            wf_src.write(src + '\n')
            wf_tgt.write(tg + '\n')
    wf_tgt.close()
    wf_src.close()

def convert_dataset_test(pair_src, pair_tgt, b):
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
        if len(tgt) == 0:
            continue
        src = process_src(cur_triples)
        wf_src.write(src + '\n')
        wf_tgt.write(json.dumps(tgt) + '\n')
    wf_tgt.close()
    wf_src.close()



outdir = 'data/webnlg_cat'
b = Benchmark()
files = select_files('webnlg_challenge_2017/train')
b.fill_benchmark(files)

pair_train_src = os.path.join(outdir, "train.source")
pair_train_tgt = os.path.join(outdir, "train.target")
convert_dataset(pair_train_src, pair_train_tgt, b)

b = Benchmark()
files = select_files('webnlg_challenge_2017/dev')
b.fill_benchmark(files)

pair_valid_src = os.path.join(outdir, "val.source")
pair_valid_tgt = os.path.join(outdir, "val.target")
convert_dataset(pair_valid_src, pair_valid_tgt, b)


b = Benchmark()
files = [('webnlg_challenge_2017/test', 'testdata_with_lex.xml')]
b.fill_benchmark(files)

pair_valid_src = os.path.join(outdir, "test.source")
pair_valid_tgt = os.path.join(outdir, "test.target")
convert_dataset_test(pair_valid_src, pair_valid_tgt, b)