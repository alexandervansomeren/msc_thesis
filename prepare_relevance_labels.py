import os
import gensim
import nltk
import pickle


def prepare_relevance_labels(output_fname='rel_labels.p', folder='original_articles/'):
    source_dict = {}  # maps article filename to source
    docs = []  # list with documents
    doc_names = []  # doc names with same index as docs

    for subdir, dirs, files in os.walk('./raw_data/' + folder):
        files = [fi for fi in files if fi.endswith(".txt")]
        for file in files:
            path = os.path.join(subdir, file)
            folder_name = subdir.split('/')[-1]
            fname = file[:-4]
            source_dict[fname] = subdir.split('/')[-1]
            with open(path, 'r') as f:
                docs.append(f.read())
            doc_names.append(fname)

    tokenized = []
    for doc in docs:
        tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
        tokenized.append(tokens)

    print(len(docs))

    # remove words that appear only once

    # frequency = defaultdict(int)
    # for text in docs:
    #     for token in text:
    #         frequency[token] += 1
    # docs = [[token for token in text if frequency[token] > 1]
    #          for text in docs]


    # create a Gensim dictionary from the texts
    # dictionary = corpora.Dictionary(docs)

    bm25 = gensim.summarization.bm25.BM25(tokenized)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

    bm25_scores = []
    sorted_bm25_indices = []
    for doc in tokenized:
        temp_bm25_scores = bm25.get_scores(doc, average_idf)
        sorted_indices = sorted(range(len(temp_bm25_scores)), key=lambda x: temp_bm25_scores[x], reverse=True)
        bm25_scores.append(temp_bm25_scores)
        sorted_bm25_indices.append(sorted_indices)

    with open(output_fname, 'wb') as f:
        pickle.dump([source_dict, docs, doc_names, tokenized, bm25_scores, sorted_bm25_indices], f)


if __name__ == "__main__":
    prepare_relevance_labels()
