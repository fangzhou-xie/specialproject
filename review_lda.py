import sqlite3 as lite
from random import choice

import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

spacy.prefer_gpu()

data_file = 'data/reviews.sqlite'


class LDAReview():
    """wrap reviews and process by LDA"""

    def __init__(self, id_reviews):
        # self.data = id_reviews
        self.sentences, self.id2oldid, self.id2review, self.reviewid2docid = \
            preprocess(id_reviews)
        print('Random select documents: {}'.format(choice(self.sentences)))
        self.dictionary = dictionary = Dictionary(self.sentences)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.bow_corpus = [dictionary.doc2bow(doc) for doc in self.sentences]

    def run(self, num_topics=10, passes=2, n_workers=20):
        model = LdaMulticore(self.bow_corpus, num_topics=num_topics,
                             id2word=self.dictionary,
                             passes=passes, workers=n_workers)
        for idx, topic in model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))


def preprocess(id_reviews):
    nlp = spacy.load('en_core_web_sm', disable=["tagger"])
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    ids, reviews = zip(*id_reviews)
    reviewnew = []
    [reviewnew.append(' '.join(i.split())) for i in reviews]

    id2oldid, id2review, reviewid2docid = {}, {}, {}
    sentences = []
    parsed_docs = [[[t.lemma_.lower() for t in sen if not (t.is_stop or t.is_punct or len(t.lemma_.strip()) == 0)]
                    for sen in doc.sents]
                   for doc in nlp.pipe(reviews, disable=['tagger'])]
    sen_id = 0
    for i, sens in enumerate(parsed_docs):
        id2oldid[i] = ids[i]
        id2review[i] = reviews[i]
        sentences += sens
        for sen in sens:
            if len(sen) > 0:
                reviewid2docid[sen_id] = i
                sen_id += 1
    return sentences, id2oldid, id2review, reviewid2docid


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
