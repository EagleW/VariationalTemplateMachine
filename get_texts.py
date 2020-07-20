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


def convert_dataset(b, ids):
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
            ids.extend([h,t])
    

outdir = 'data/webnlg_tag'

b = Benchmark()
ids = []

pair_valid_src = os.path.join(outdir, "ids.json")
files = select_files('webnlg_challenge_2017/dev')
b.fill_benchmark(files)
convert_dataset(b, ids)


b = Benchmark()
files = [('webnlg_challenge_2017/test', 'testdata_with_lex.xml')]
b.fill_benchmark(files)
convert_dataset(b, ids)


wf_tgt = open(pair_valid_src, 'w')
json.dump(ids, wf_tgt, indent = 4)
wf_tgt.close()