import json
import os
import pdb
import random
import sqlite3 as lite
from operator import itemgetter

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
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.sentences]
        doc_len = len(self.bow_corpus)  # should equal to length of sentences
        ids = random.shuffle(doc_len)
        self.shuffled2senid = {v: k for k, v in enumerate(ids)}

        test_ids = ids[:round(doc_len * test_ratio)]
        train_ids = ids[round(doc_len * test_ratio):]

        self.bows = bow_corpus
        self.train_bows = [bow_corpus[i] for i in train_ids]
        self.test_bows = [bow_corpus[i] for i in test_ids]

    def cv(self):
        'cross validate the hyper-parameters. log_perplexity smaller the better'
        best_ppl = 1e6
        best_para = {'topic': 10,
                     'ps': 2,
                     'alpha': 0.05,
                     'beta': 0.05}
        for topic in [10, 20, 30]:
            for ps in [2, 4]:
                for alpha in [0.05, 0.1, 0.5, 1, 5, 10]:
                    for beta in [0.05, 0.1, 0.5, 1, 5, 10]:
                        model = run(self.train_bows, self.dictionary,
                                    num_topics=topic, passes=ps,
                                    alpha=alpha, eta=beta)
                        log_ppl = model.log_perplexity(self.test_bows)
                        if log_ppl < best_ppl:
                            best_ppl = log_ppl
                            best_para['alpha'] = alpha
                            best_para['beta'] = beta
                            best_para['ps'] = ps
                            best_para['topic'] = topic
        alpha, beta, ps, topic = [best_para[i]
                                  for i in sorted(best_para, key=itemgetter(0))]
        model = run(self.bows, self.dictionary,
                    num_topics=topic, passes=ps,
                    alpha=alpha, eta=beta)
        total_topics = {k: v for k, v in model.print_topics(-1)}
        topic_weight = {self.senid2docid[self.shuffled2senid[i]]: model[j]
                        for i, j in enumerate(self.bows)}
        # save the topics, weights for docs, and document id to the orginal id
        with open('js_topics.json', 'w') as f:
            json.dump(total_topics, f)
        with open('js_weight.json', 'w') as f:
            json.dump(topic_weight, f)
        with open('js_docid2oldid', 'w') as f:
            json.dump(self.docid2oldid, f)


def run(self, bow_corpus, dictionary,
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
            if len(sen) > 0:
                senid2docid[sen_id] = i
                sen_id += 1
    return sentences, docid2oldid, docid2review, senid2docid


def loaddata():
    conn = lite.connect(data_file)
    c = conn.cursor()
    result = c.execute("select id, comments from reviews;").fetchall()
    return result


def main():
    id_reviews = loaddata()
    lda = LDAReview(id_reviews)
    lda.run()
    # pdb.set_trace()
    pass


if __name__ == "__main__":
    main()
