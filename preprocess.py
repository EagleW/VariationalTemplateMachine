from benchmark_reader import Benchmark
import os, json
from collections import Counter
from benchmark_reader import select_files
from nltk import word_tokenize
from tqdm import tqdm

def process_src(triples, majority):
    src = 'name:' + majority + '\t'
    # count = 2+ str(count) 
    for h, r, t in triples:
        if h != majority:
            src += h + '_' + r + ':' + t + '\t'
        else:
            src += r + ':' + t + '\t'
        # count += 1
    return src



def process_tgt(entities, texts):
    majority = Counter()
    for text in texts:
        for e in entities:
            if e in text.lex:
                majority[text.lex] += 1
    if len(majority) == 0:
        return 0
    text = majority.most_common(1)[0][0]
    tgt = word_tokenize(text)
    text = ' '.join(tgt) + ' <eos>|||\n'
    for e in entities:
        text = text.replace(e, entities[e])
    return  text

def process_tgt_test(entities, texts):
    majority = Counter()
    for text in texts:
        for e in entities:
            if e in text.lex:
                majority[text.lex] += 1
    if len(majority) == 0:
        return 0
    texts = []
    num = majority.most_common(1)[0][1]
    for text, i in majority.most_common():
        if i == num:
            tgt = word_tokenize(text)
            text = ' '.join(tgt)
            for e in entities:
                text = text.replace(e, entities[e])
            texts.append(text)
        else:
            break
    return  texts

def convert_dataset(pair_src, pair_tgt, b):
    wf_src = open(pair_src, 'w')
    wf_tgt = open(pair_tgt, 'w')
    for entry in tqdm(b.entries):
        entities = {}
        majority = Counter()

        triples = entry.list_triples()
        if len(triples) == 0:
            continue
        cur_triples = []
        for triple in triples:
            h, r, t = triple.split(' | ')
            h = h.replace('"', '')
            t = t.replace('"', '')
            r = r.replace('"', '')
            cur_triples.append((h,r.replace(' ', '_'),t))
            majority[h] += 1
            entities[' '.join(word_tokenize(h.replace('_', ' ')))] = h
            entities[' '.join(word_tokenize(r.replace('_', ' ')))] = r
        tgt = process_tgt(entities, entry.lexs)
        if tgt == 0:
            continue
        src = process_src(cur_triples, majority.most_common(1)[0][0])
        wf_src.write(src + '\n')
        wf_tgt.write(tgt)
    wf_tgt.close()
    wf_src.close()

def convert_dataset_test(pair_src, pair_tgt, b):
    wf_src = open(pair_src, 'w')
    wf_tgt = open(pair_tgt, 'w')
    for entry in tqdm(b.entries):
        entities = {}
        all_e = []
        majority = Counter()

        triples = entry.list_triples()
        if len(triples) == 0:
            continue
        cur_triples = []
        for triple in triples:
            h, r, t = triple.split(' | ')
            h = h.replace('"', '')
            t = t.replace('"', '')
            r = r.replace('"', '')
            cur_triples.append((h,r.replace(' ', '_'),t))
            majority[h] += 1
            entities[' '.join(word_tokenize(h.replace('_', ' ')))] = h
            entities[' '.join(word_tokenize(r.replace('_', ' ')))] = r
            all_e.append(h)
            all_e.append(r)
        tgt = process_tgt_test(entities, entry.lexs)
        if tgt == 0:
            continue
        src = process_src(cur_triples, majority.most_common(1)[0][0])
        wf_src.write(src + '\n')
        wf_tgt.write(json.dumps([tgt, all_e]) + '\n')
    wf_tgt.close()
    wf_src.close()


outdir = 'data/webnlg'
b = Benchmark()
files = select_files('webnlg_challenge_2017/train')
b.fill_benchmark(files)

pair_train_src = os.path.join(outdir, "pair_src.train")
pair_train_tgt = os.path.join(outdir, "pair_tgt.train")
convert_dataset(pair_train_src, pair_train_tgt, b)

b = Benchmark()
files = select_files('webnlg_challenge_2017/dev')
b.fill_benchmark(files)

pair_valid_src = os.path.join(outdir, "pair_src.valid")
pair_valid_tgt = os.path.join(outdir, "pair_tgt.valid")
convert_dataset(pair_valid_src, pair_valid_tgt, b)

b = Benchmark()
files = [('webnlg_challenge_2017/test', 'testdata_unseen_with_lex.xml'), ('webnlg_challenge_2017/test', 'testdata_with_lex.xml')]
b.fill_benchmark(files)

pair_valid_src = os.path.join(outdir, "src_test.txt")
pair_valid_tgt = os.path.join(outdir, "test_refs.txt")
convert_dataset_test(pair_valid_src, pair_valid_tgt, b)