from collections import defaultdict

import numpy as np
import os
import pickle

import networkx as nx
from sklearn.model_selection import KFold


def nonedges(G, u):  # a generator with (u,v) for every non neighbor v
    for v in nx.non_neighbors(G, u):
        yield (u, v)


def links_per_doc_creator(links, links_ixs):
    temp_links_per_doc = defaultdict(set)
    for link_ix in links_ixs:
        temp_links_per_doc[links[link_ix][0]].add(links[link_ix][1])
    return temp_links_per_doc


def read_and_sample_data(data_folder, sample_size, docs, doc_names, links, sample_seed, strategy='binned'):
    """
    Obtains data from "data.tmp" folder
    :param data_folder: folder of the data
    :param sample_size: if sample_size is bigger than zero, randomly sample sample_size documents
    """

    # global docs  # list with documents
    # global doc_names  # doc names with same index as docs
    # global links
    # global sample_seed

    def read_data():
        data_path = os.path.join(os.getcwd(), 'data.tmp', data_folder)
        with open(os.path.join(data_path, 'texts.txt'), 'r', encoding='utf8') as f:
            for line in f:
                line = line.split(' ')
                doc_names.append(line[0])
                docs.append(' '.join(line[1:]))
        print('    Read in docs.. Reading references..')
        # doc_names_set = set(doc_names)  # ~300ms
        doc_names_indexes = {k: v for v, k in enumerate(doc_names)}
        links = []
        with open(os.path.join(data_path, 'links.txt'), 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split(' ')
                link = []
                for l in line[1:]:
                    try:
                        link.append(doc_names_indexes[l])
                    except KeyError:
                        pass
                links.append(link)
                # links.append([doc_names.index(l) for l in
                #               line[1:] if l in doc_names_set])  # only add links that are present in doc_names
        return docs, doc_names, links

    if strategy == "expand":
        data_path = os.path.join(os.getcwd(), 'data.tmp', data_folder)
        data_cache = os.path.join(data_path, '_'.join(['', strategy, str(sample_size), str(sample_seed)])) + '.p'
        if os.path.exists(data_cache):
            return pickle.load(open(data_cache, 'rb'))
        else:
            print("    Reading raw data")
            docs, doc_names, links = read_data()
            print("    Done.")
            start_sample_size = int(sample_size / 100)
            sampled_links = []
            sampled_doc_ixs = set(np.random.choice(len(docs), start_sample_size, replace=False))
            new_sample = sampled_doc_ixs.copy()
            while True:
                temp_sample = set()
                for source in new_sample:
                    # sampled_doc_ixs.add(source)
                    for target in links[source]:
                        sampled_links.append((source, target))
                        sampled_doc_ixs.add(target)
                        temp_sample.add(target)
                        if len(sampled_doc_ixs) == sample_size:
                            sampled_doc_ixs = list(sampled_doc_ixs)
                            sampled_doc_names = [doc_names[ix] for ix in sampled_doc_ixs]
                            s_doc_names_ixs = {k: v for v, k in enumerate(sampled_doc_names)}
                            sampled_links = [(s_doc_names_ixs[doc_names[s]], s_doc_names_ixs[doc_names[t]]) for s, t in
                                             list(set(sampled_links))]
                            sampled_docs = [docs[ix] for ix in sampled_doc_ixs]
                            with open(data_cache, 'wb') as f:
                                pickle.dump((sampled_docs, sampled_doc_names, sampled_links), f)
                            return sampled_docs, sampled_doc_names, sampled_links  # docs, doc_names, links
                new_sample = temp_sample.difference(new_sample)
                print("    Current sample size: " + str(len(sampled_doc_ixs)))

    # elif strategy == "binned":
    #     docs, doc_names, links = read_data()
    #     if 0 < sample_size < len(doc_names):
    #         numbers_of_links = [len(l) for l in links]
    #         bin_limits = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99999)]
    #         doc_indices = []
    #
    #         per_bin_sample_size = int(sample_size / len(bin_limits))
    #
    #         for bin_lower_limit, bin_upper_limit in bin_limits:
    #
    #             # don't sample no links
    #             indices_in_bin = [i for i, v in enumerate(numbers_of_links) if (bin_upper_limit >= v > bin_lower_limit)]
    #
    #             # print(bin_lower_limit, len(indices_in_bin))
    #             if per_bin_sample_size > len(indices_in_bin):
    #                 print("per_bin_sample_size > len(indices_in_bin) in bin", bin_lower_limit)
    #                 choices = indices_in_bin
    #             else:
    #                 choices = np.random.choice(indices_in_bin, per_bin_sample_size, replace=False)
    #             doc_indices.extend(choices)
    #
    #         target_doc_indices = []
    #         print(len(doc_indices))
    #         for doc_index in doc_indices:  # add target docs from links
    #             target_doc_indices.extend([doc_names.index(ti) for ti in links[doc_index]])
    #         doc_indices.extend(target_doc_indices)
    #
    #         doc_indices = list(set(doc_indices))  # make sure that there are no duplicates.
    #
    #         print('Initial sample size: ', sample_size)
    #         print('Final number of docs (after adding target documents from links):', len(doc_indices))
    #         docs = [docs[choice] for choice in doc_indices]
    #         doc_names = [doc_names[choice] for choice in doc_indices]
    #         links = [links[choice] for choice in doc_indices]  # assumes same order of text and link file!

    temp_links = []
    for doc_name, links_per_node in zip(doc_names, links):
        for link in links_per_node:
            temp_links.append((doc_name, link))
    links = temp_links
    return docs, doc_names, links


def split_into_k_folds(links, k=5, strategy='random', sample_seed=2017):
    """
    Splits links into k-folds
    :param k: number of folds
    :param strategy:  use this splitting strategy
    :param sample_seed: seed for random sampling
    :return split_indices: tuple of indices
    """
    # print(links[:10])
    if strategy == 'random':
        kf = KFold(n_splits=k, random_state=sample_seed, shuffle=True)
        split_indices = kf.split(links)
        return split_indices
    elif strategy == 'random_per_source':
        pass


##### OLD ######


# def evaluate_BM25():
#     """
#     Evaluate
#     """
#     import rank_metrics
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


def get_relevance_labels(data_folder):
    """
    Prepares or loads the relevance labels using bm25
    """
    rel_labels_fname = 'relevance_labels_' + data_folder + '.p'
    global tokenized_docs
    global sorted_bm25_indices
    global doc_names
    if not os.path.isfile(rel_labels_fname):
        print('Computing relevance labels using bm25... (this takes a while, but should only be done once..)')
        import prepare_relevance_labels
        prepare_relevance_labels.prepare_relevance_labels(output_fname=rel_labels_fname, folder=data_folder)
    with open(rel_labels_fname, 'rb') as rel_lab_file:
        _, _, _, tokenized_docs, doc_names, sorted_bm25_indices = pickle.load(rel_lab_file)
        # return
