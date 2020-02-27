import json
import os
import pdb
import pickle
import random
import sqlite3 as lite
from operator import itemgetter

import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

spacy.prefer_gpu()
random.seed(1)

data_file = 'data/reviews.sqlite'
ckpt_path = './ckpt/'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


class LDAReview():
    """wrap reviews and process by LDA"""

    def __init__(self, id_reviews, test_ratio):
        # self.data = id_reviews
        self.sentences, self.docid2oldid, self.docid2review, self.senid2docid = \
            preprocess(id_reviews)
        print('Random select documents: {}'.format(
            random.choice(self.sentences)))
        self.dictionary = dictionary = Dictionary(self.sentences)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.bow_corpus = bow_corpus = [
            dictionary.doc2bow(doc) for doc in self.sentences]
        doc_len = len(self.bow_corpus)  # should equal to length of sentences
        ids = list(range(doc_len))
        random.shuffle(ids)
        self.shuffled2senid = {v: k for k, v in enumerate(ids)}

        test_ids = ids[:round(doc_len * test_ratio)]
        train_ids = ids[round(doc_len * test_ratio):]

        self.train_bows = [bow_corpus[i] for i in train_ids]
        self.test_bows = [bow_corpus[i] for i in test_ids]

    def cv(self):
        'cross validate the hyper-parameters. log_perplexity smaller the better'
        best_ppl = 1e6
        best_para = {'topic': 30,
                     'ps': 2,
                     'alpha': 0.5,
                     'beta': 5}
        # for topic in [20, 30]:
        #     for ps in [2, 4]:
        #         for alpha in [0.05, 0.1, 0.5, 1, 5]:
        #             for beta in [0.05, 0.1, 0.5, 1, 5]:
        #                 model = run(self.train_bows, self.dictionary,
        #                             num_topics=topic, passes=ps,
        #                             alpha=alpha, eta=beta)
        #                 log_ppl = model.log_perplexity(self.test_bows)
        #                 if log_ppl < best_ppl:
        #                     best_ppl = log_ppl
        #                     best_para['alpha'] = alpha
        #                     best_para['beta'] = beta
        #                     best_para['ps'] = ps
        #                     best_para['topic'] = topic
        alpha, beta, ps, topic = [best_para[i]
                                  for i in sorted(best_para, key=itemgetter(0))]
        print(best_para)
        model = run(self.bow_corpus, self.dictionary,
                    num_topics=topic, passes=ps,
                    alpha=alpha, eta=beta)
        total_topics = {k: v for k, v in model.print_topics(-1, num_words=20)}
        # pdb.set_trace()
        topic_weight = {self.senid2docid[self.shuffled2senid[i]]: model[j]
                        for i, j in enumerate(self.bow_corpus)}
        # save the topics, weights for docs, and document id to the orginal id
        with open('js_topics.pk', 'wb') as f:
            pickle.dump(total_topics, f)
        with open('js_weight.pk', 'wb') as f:
            pickle.dump(topic_weight, f)
        with open('js_docid2oldid.pk', 'wb') as f:
            pickle.dump(self.docid2oldid, f)


def run(bow_corpus, dictionary,
        num_topics=10, passes=2, n_workers=20, **kwargs):
    # best_ppl = 1e8
    model = LdaMulticore(bow_corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         passes=passes,
                         workers=n_workers,
                         random_state=0,
                         minimum_probability=0,
                         **kwargs)
    # for idx, topic in model.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))
    return model


def preprocess(id_reviews):
    nlp = spacy.load('en_core_web_sm', disable=["tagger"])
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    ids, reviews = zip(*id_reviews)
    reviewnew = []
    [reviewnew.append(' '.join(i.split())) for i in reviews]

    docid2oldid, docid2review, senid2docid = {}, {}, {}
    sentences = []
    parsed_docs = [[[t.lemma_.lower() for t in sen if not
                     (t.is_stop or t.is_punct or len(t.lemma_.strip()) == 0)]
                    for sen in doc.sents]
                   for doc in nlp.pipe(reviews, disable=['tagger'])]
    sen_id = 0
    for i, sens in enumerate(parsed_docs):
        docid2oldid[i] = ids[i]
        docid2review[i] = reviews[i]
        sentences += sens
        for sen in sens:
            senid2docid[sen_id] = i
            sen_id += 1
    return sentences, docid2oldid, docid2review, senid2docid


def loaddata():
    conn = lite.connect(data_file)
    c = conn.cursor()
    result = c.execute("select id, comments from reviews;").fetchall()
    return result


def cvandsave():
    id_reviews = loaddata()
    lda = LDAReview(id_reviews, 0.3)
    lda.cv()
    # pdb.set_trace()
    pass


def interpret():
    with open('js_topics.pk', 'rb') as f:
        total_topics = pickle.load(f)
    with open('js_weight.pk', 'rb') as f:
        topic_weight = pickle.load(f)
    with open('js_docid2oldid.pk', 'rb') as f:
        docid2oldid = pickle.load(f)
    # pdb.set_trace()
    # print('end')
    # write topics.csv
    with open('topics.csv', 'w') as f:
        f.write('num,' + ','.join(['word{}'.format(i)
                                   for i in range(10)]) + '\n')
    for k, v in total_topics.items():
        tpweight = [i.strip() for i in v.split('+')]
        tpweight.insert(0, str(k))
        with open('topics.csv', 'a') as f:
            f.write(','.join(tpweight) + '\n')

    # write doc weights
    dt = np.dtype('int,float')
    combined_dict = {}
    for k, v in topic_weight.items():
        oldid = docid2oldid[k]
        valuepair = np.array(v, dtype=dt)
        valuepair.dtype.names = ['ind', 'weight']
        if oldid in combined_dict:
            combined_dict[oldid] += valuepair['weight']
        else:
            combined_dict[oldid] = valuepair['weight']
    pd.DataFrame.from_dict(combined_dict, orient='index').to_csv(
        'weights.csv', index=True)


def main():
    # cvandsave()
    interpret()


if __name__ == "__main__":
    main()
