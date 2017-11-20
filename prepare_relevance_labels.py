import os
import gensim
import nltk
import pickle

import time

import math


class BM25:
    def __init__(self, data_path):
        self.dictionary = gensim.corpora.Dictionary()
        self.doc_names = []
        self.source_dict = {}
        self.DF = {}
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.data_path = data_path
        self.DocLen = []
        self.build_dictionary()
        self.tfidf_generator()

    def build_dictionary(self):
        raw_data = []
        for subdir, dirs, files in os.walk(self.data_path):
            files = [fi for fi in files if fi.endswith(".txt")]
            for file in files:
                path = os.path.join(subdir, file)
                folder_name = subdir.split(os.path.sep)[-1]
                fname = file[:-4]
                self.source_dict[fname] = subdir.split(os.path.sep)[-1]
                with open(path, 'r', encoding='utf8') as f:
                    raw_data.append(
                        [word for sent in nltk.sent_tokenize(f.read()) for word in nltk.word_tokenize(sent)])
                self.doc_names.append(fname)
        self.dictionary.add_documents(raw_data)

    def tfidf_generator(self, base=math.e):
        total_doc_len = 0
        for subdir, dirs, files in os.walk(self.data_path):
            files = [fi for fi in files if fi.endswith(".txt")]
            for file in files:
                path = os.path.join(subdir, file)
                with open(path, 'r', encoding='utf8') as f:
                    doc = [word for sent in nltk.sent_tokenize(f.read()) for word in nltk.word_tokenize(sent)]
                total_doc_len += len(doc)
                self.DocLen.append(len(doc))
                bow = dict([(term, freq * 1.0 / len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
                for term, tf in bow.items():
                    if term not in self.DF:
                        self.DF[term] = 0
                    self.DF[term] += 1
                self.DocTF.append(bow)
                self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] + 0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = total_doc_len / self.N

    def bm25_score(self, query=[], k1=1.5, b=0.75):
        query_bow = self.dictionary.doc2bow(query)
        scores = []
        for idx, doc in enumerate(self.DocTF):
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms:
                upper = (doc[term] * (k1 + 1))
                below = ((doc[term]) + k1 * (1 - b + b * doc_terms_len / self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def tfidf(self):
        tfidf = []
        for doc in self.DocTF:
            doc_tfidf = [(term, tf * self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def items(self):
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items.sort()
        return items


def prepare_relevance_labels(output_fname='rel_labels.p', folder='original_articles'):
    source_dict = {}  # maps article filename to source
    docs = []  # list with documents
    doc_names = []  # doc names with same index as docs

    data_path = os.path.join(os.getcwd(), 'raw_data.tmp', folder)
    for subdir, dirs, files in os.walk(data_path):
        files = [fi for fi in files if fi.endswith(".txt")]
        for file in files:
            path = os.path.join(subdir, file)
            folder_name = subdir.split(os.path.sep)[-1]
            fname = file[:-4]
            source_dict[fname] = subdir.split(os.path.sep)[-1]
            with open(path, 'r', encoding='utf8') as f:
                docs.append(f.read())
            doc_names.append(fname)

    tokenized = []
    for doc in docs:
        tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
        tokenized.append(tokens)

    print(len(docs))
    print("Computing BM25...")
    bm25 = BM25(data_path)
    print("Done computing bm25.")

    print("Computing relative scores...")
    bm25_scores = []
    sorted_bm25_indices = []
    len_tokenized = len(tokenized)
    get_scores_time = 0.0
    sort_time = 0.0
    for i, doc in enumerate(tokenized):
        start_score = time.time()
        temp_bm25_scores = bm25.bm25_score(doc)
        end_score = time.time()
        get_scores_time += end_score - start_score
        sorted_indices = sorted(range(len(temp_bm25_scores)), key=lambda x: temp_bm25_scores[x], reverse=True)
        sort_time += time.time() - end_score
        sorted_bm25_indices.append(sorted_indices[:11])
        if (i % 100) == 0:
            print(str(i) + ' / ' + str(len_tokenized))
            print("Total score time:", get_scores_time, 'seconds.')
            print("Total sort time:", sort_time, 'seconds.')
            with open(output_fname, 'wb') as f:
                pickle.dump([source_dict, docs, doc_names, tokenized, bm25_scores, sorted_bm25_indices], f)
        with open(output_fname, 'wb') as f:
            pickle.dump([source_dict, docs, doc_names, tokenized, bm25_scores, sorted_bm25_indices], f)


if __name__ == "__main__":
    prepare_relevance_labels()
