import pickle
import os
import argparse

from collections import defaultdict

import shutil

import doc2vec

import rank_metrics
import numpy as np
import nltk
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # cancel optimization complaints by tensorflow


def read_and_sample_data(sample_size=0):
    """
    Obtains data from "raw_data.tmp" folder
    :param sample_size: if sample_size is bigger than zero, randomly sample sample_size documents
    """
    global docs  # list with documents
    global doc_names  # doc names with same index as docs
    global links
    global sample_seed

    data_path = os.path.join(os.getcwd(), 'raw_data.tmp', raw_data_folder)
    with open(os.path.join(data_path, 'texts.txt'), 'r', encoding='utf8') as f:
        for line in f:
            line = line.split(' ')
            doc_names.append(line[0])
            docs.append(' '.join(line[1:]))

    doc_names_set = set(doc_names)
    links = []
    with open(os.path.join(data_path, 'links.txt'), 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split(' ')
            links.append([l for l in line[1:] if l in doc_names_set])  # only add links that are present in doc_names

    # if args.remove_docs_without_links:
    #     rm_list = reversed([i for i, numbers_of_links in enumerate([len(l) for l in links]) if numbers_of_links == 0])
    #     for rm_ix in rm_list:
    #         del (doc_names[rm_ix])
    #         del (links[rm_ix])
    #         del (docs[rm_ix])
    #     print(len(doc_names))

    if 0 < sample_size < len(doc_names):
        np.random.seed(sample_seed)
        # choices = np.random.choice(len(docs), sample_size, replace=False)
        # docs = [docs[choice] for choice in choices]
        # doc_names = [doc_names[choice] for choice in choices]
        # links = [links[choice] for choice in choices]  # assumes same order of text and link file!

        # sample using buckets from len(..

        numbers_of_links = [len(l) for l in links]
        bin_limits = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99999)]
        doc_indices = []

        per_bin_sample_size = int(sample_size / len(bin_limits))

        for bin_lower_limit, bin_upper_limit in bin_limits:

            # don't sample no links
            indices_in_bin = [i for i, v in enumerate(numbers_of_links) if (bin_upper_limit >= v > bin_lower_limit)]

            # print(bin_lower_limit, len(indices_in_bin))
            if per_bin_sample_size > len(indices_in_bin):
                print("per_bin_sample_size > len(indices_in_bin) in bin", bin_lower_limit)
                choices = indices_in_bin
            else:
                choices = np.random.choice(indices_in_bin, per_bin_sample_size, replace=False)
            doc_indices.extend(choices)

        target_doc_indices = []
        print(len(doc_indices))
        for doc_index in doc_indices:  # add target docs from links
            target_doc_indices.extend([doc_names.index(ti) for ti in links[doc_index]])
        doc_indices.extend(target_doc_indices)

        doc_indices = list(set(doc_indices))  # make sure that there are no duplicates.

        print('Initial sample size: ', sample_size)
        print('Final number of docs (after adding target documents from links):', len(doc_indices))
        docs = [docs[choice] for choice in doc_indices]
        doc_names = [doc_names[choice] for choice in doc_indices]
        links = [links[choice] for choice in doc_indices]  # assumes same order of text and link file!

        # Based on some inverse of PDF
        #
        # counter = Counter(numbers_of_links).items()
        # counts = [c[0] for c in counter]
        # sample_pdf = np.array([1 - c[1] / len(numbers_of_links) for c in counter])
        # sample_pdf /= sum(sample_pdf)
        # samples_per_count = np.random.choice(len(counts), sample_size, replace=True)

    temp_links = []
    for doc_name, links_per_node in zip(doc_names, links):
        for link in links_per_node:
            temp_links.append((doc_name, link))
    links = temp_links


def split_into_k_folds(k=5, strategy='random'):
    """
    Splits links into k-folds
    :param strategy:  use this splitting strategy
    :param k: number of folds
    :return split_indices: tuple of indices
    """
    print(links[:10])
    if strategy == 'random':
        kf = KFold(n_splits=k, random_state=sample_seed, shuffle=True)
        split_indices = kf.split(links)
        return split_indices
    elif strategy == 'random_per_source':
        pass


def get_relevance_labels():
    """
    Prepares or loads the relevance labels using bm25
    """
    global tokenized_docs
    global sorted_bm25_indices
    global doc_names
    if not os.path.isfile(rel_labels_fname):
        print('Computing relevance labels using bm25... (this takes a while, but should only be done once..)')
        import prepare_relevance_labels
        prepare_relevance_labels.prepare_relevance_labels(output_fname=rel_labels_fname, folder=raw_data_folder)
    with open(rel_labels_fname, 'rb') as rel_lab_file:
        _, _, _, tokenized_docs, doc_names, sorted_bm25_indices = pickle.load(rel_lab_file)
        # return


def train_paragraph_vectors():
    """
    Train paragraph vectors
    """

    print('Initializing and training paragraph vectors')
    d2v = doc2vec.Doc2Vec(**params)

    if params.architecture == 'pvdm':
        d2v.fit(tokenized_docs)
    elif params.architecture == 'pvprior':
        d2v.fit(tokenized_docs, sorted_bm25_indices)
    d2v.save(doc2vec_model_folder)
    return d2v


def train_paragraph_vectors_gensim():
    global tokenized_docs
    # print(tokenized_docs[0])
    model = gensim.models.Doc2Vec(tokenized_docs,
                                  size=args.emb_size_d,
                                  window=params['window_size'],
                                  min_count=5,
                                  max_vocab_size=params['vocabulary_size'],
                                  alpha=params['learning_rate'],
                                  iter=args.iterations,
                                  workers=4)
    model.save(doc2vec_model_folder)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model


# def evaluate_BM25():
#     """
#     Evaluate
#     """
#     print("Evaluating embeddings agains BM25")
#     neigh = NearestNeighbors(n_neighbors=10, algorithm='brute', metric=distance_measure)
#     neigh.fit(embeddings)
#     sorted_distances_indices = []
#     for embedding in embeddings:
#         sorted_distances_indices.append(neigh.kneighbors(embedding.reshape(1, -1), 10, return_distance=False))
#
#     results = []
#     hits = []
#     for doc_index, sorted_distance_indices in enumerate(sorted_distances_indices):
#         relevance_set = set(sorted_bm25_indices[doc_index][1:11])
#         hit = np.array([ix in relevance_set for ix in sorted_distance_indices[0][:10]], dtype=int)
#         average_precision = rank_metrics.average_precision(hit)
#         ndcg_at_10 = rank_metrics.ndcg_at_k(hit, 10)
#         hits.append(hit)
#         results.append({
#             'average_precision': average_precision,
#             'ndcg_at_10': ndcg_at_10,
#         })
#         if doc_index % 1000 == 0:
#             with open(os.path.join(exp_dir, 'results.p'), 'wb') as f:
#                 pickle.dump(results, f)
#
#     with open(os.path.join(exp_dir, 'results.p'), 'wb') as f:
#         pickle.dump(results, f)


def links_per_doc_creator(links_ixs):
    temp_links_per_doc = defaultdict(set)
    for link_ix in links_ixs:
        temp_links_per_doc[links[link_ix][0]].add(links[link_ix][1])
    return temp_links_per_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('-algorithm',
                        type=str,
                        default='pvdm',
                        choices=['pvdm', 'pvprior', 'doc2vec-gensim', 'random'],
                        help='Sets the algorithm you want to use')
    parser.add_argument('-data',
                        type=str,
                        default='aminer_org_v1',
                        choices=['aminer_org_v1'],
                        help='Name of data folder (placed in raw_data.tmp)')
    parser.add_argument('-sample',
                        type=int,
                        default=50000,
                        help='Sample from data set')
    parser.add_argument('-iterations',
                        type=int,
                        default=5,
                        help='Number of iterations over data for algorithm')
    parser.add_argument('-emb_size_w',
                        type=int,
                        default=300,
                        help='Embedding size of words')
    parser.add_argument('-emb_size_d',
                        type=int,
                        default=300,
                        help='Embedding size of documents')
    parser.add_argument('-dist_measure',
                        type=str,
                        default='cosine',
                        choices=['cosine', 'pvprior'],
                        help='Distance measure for the evaluation')
    parser.add_argument('-seed',
                        type=int,
                        default=0,
                        help='Seed for data sampling')
    parser.add_argument('-remove_docs_without_links',
                        type=bool,
                        default=True,
                        help='Remove documents that do not have any links to other documents in dataset')

    args = parser.parse_args()
    exp_name = '_'.join([str(val) for val in vars(args).values()])
    exp_dir = os.path.join('experiments', exp_name)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    else:
        inp = input("Experiment folder already exists, press x to exit. ")
        if inp == 'x':
            exit(1)
        inp = input("Clean experiment folder? press y to clean. ")
        if inp == 'y':
            shutil.rmtree(exp_dir)
            os.mkdir(exp_dir)

    print(args)
    raw_data_folder = args.data
    rel_labels_fname = 'relevance_labels_' + raw_data_folder + '.p'
    doc2vec_model_folder = os.path.join(exp_dir, 'model')
    architecture = args.algorithm

    n_folds = 5.

    distance_measure = args.dist_measure
    sample_seed = args.seed

    docs = []
    doc_names = []
    links = []

    print("Reading and sampling data...")
    read_and_sample_data(args.sample)

    tokenized_docs = []
    for doc in docs:
        tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
        tokenized_docs.append(tokens)
    del docs

    params = {
        'batch_size': 128,
        'window_size': 8,
        'concat': True,
        'architecture': args.algorithm,
        'embedding_size_w': args.emb_size_w,
        'embedding_size_d': args.emb_size_d,
        'vocabulary_size': 50000,
        'document_size': len(tokenized_docs) * ((n_folds - 1) / n_folds),
        'loss_type': 'sampled_softmax_loss',
        'n_neg_samples': 64,
        'optimize': 'Adagrad',
        'learning_rate': 1.0,
        'n_steps': args.iterations * len(tokenized_docs) * ((n_folds - 1) / n_folds)
    }

    if args.algorithm == "doc2vec-gensim":
        import gensim

        tokenized_docs = [gensim.models.doc2vec.TaggedDocument(toks, [i]) for i, toks in enumerate(tokenized_docs)]

        params.update(vars(args))

        print("Training paragraph vectors (gensim)")
        d2v = train_paragraph_vectors_gensim()
        print("Trained paragraph vectors")

    with open(os.path.join(exp_dir, 'params.p'), 'wb') as params_file:
        pickle.dump(params, params_file)

    print('Starting', int(n_folds), '-fold runs')

    for fold_number, split in enumerate(split_into_k_folds(int(n_folds))):

        fold_folder = os.path.join(exp_dir, 'fold_' + str(fold_number))
        if not os.path.exists(fold_folder):
            os.mkdir(fold_folder)

        left_in_links_ixs = split[0]
        left_out_links_ixs = split[1]
        print(left_out_links_ixs)

        print('In fold ' + str(fold_number) + '. Saving results in ', fold_folder)
        print('Writing ground_truth file for trec_eval.')

        links_per_doc = links_per_doc_creator(left_out_links_ixs)
        # print(links_per_doc)
        with open(os.path.join(fold_folder, 'trec_rel_file.tmp'), 'w') as trec_test_f:
            for doc, temp_links in links_per_doc.items():
                for temp_link in temp_links:
                    trec_test_f.write(' '.join([doc, '0', temp_link, '1']) + '\n')
        print('Done.')

        if args.algorithm == 'pvdm' or args.algorithm == 'pvprior':
            print("Not implemented yet for new experiment environment! -- Training paragraph vectors ")
            # d2v = train_paragraph_vectors()
            # print("Trained paragraph vectors")
            # params = d2v.get_params()
            # print("Loading embeddings")
            # with d2v.sess.as_default():
            #     embeddings = d2v.doc_embeddings  # .eval()
        elif args.algorithm == 'doc2vec-gensim':
            print("Writing results")
            # embeddings = d2v.docvecs
            with open(os.path.join(fold_folder, 'trec_top_file.tmp'), 'w') as trec_rel_f:
                for i, doc_name in enumerate(doc_names):
                    # most_similar_indices = [res[0] for res in ]
                    for s, msi in enumerate(d2v.docvecs.most_similar(i, topn=100)):
                        trec_rel_f.write(
                            ' '.join([doc_name, '0', doc_names[msi[0]], str(msi[1]), '0', exp_name]) + '\n')
        elif args.algorithm == 'random':
            print('Generating random results.')
            doc_names_set = set(doc_names)
            # print("doc_names contains duplicates?", len(doc_names_set) < len(doc_names))
            with open(os.path.join(fold_folder, 'trec_top_file.tmp'), 'w') as trec_rel_f:
                for i, doc_name in enumerate(doc_names):
                    temp_set = doc_names_set.difference({doc_name})
                    most_similar = np.random.choice(list(temp_set), 100, replace=False)
                    for s, doc in enumerate(most_similar):
                        trec_rel_f.write(' '.join([doc_name, '0', doc, str(s), '0', exp_name]) + '\n')
            print('Done.')
        print('--------------------')
    # query_id, iter, docno, rank, sim, run_id  delimited by spaces
