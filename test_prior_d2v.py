import pickle
import os

import time

import doc2vec
import scipy.spatial.distance

import rank_metrics
import numpy as np

import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # cancel optimization complaints by tensorflow

raw_data_folder = 'aminer_org_v1'  # aminer_org_v1
rel_labels_fname = 'relevance_labels_' + raw_data_folder + '.p'
architecture = 'pvprior'  # pvdm
doc2vec_model_folder = 'd2v_model_' + raw_data_folder + '_' + architecture

distance_measure = 'cosine'

with open(rel_labels_fname, 'rb') as f:
    source_dict, docs, doc_names, tokenized, bm25_scores, sorted_bm25_indices = pickle.load(f)
#
# d2v = doc2vec.Doc2Vec(batch_size=128,
#                       window_size=8,
#                       prior_sample_size=10,
#                       concat=True,
#                       architecture=architecture,
#                       embedding_size_w=300,  # word embedding size
#                       embedding_size_d=300,  # document embeding size
#                       vocabulary_size=50000,
#                       document_size=len(docs),
#                       loss_type='sampled_softmax_loss',
#                       n_neg_samples=64,
#                       optimize='Adagrad',
#                       learning_rate=1.0,
#                       n_steps=124001  # 100001
#                       )
#
# d2v.fit(tokenized, sorted_bm25_indices)
#
# d2v.save(doc2vec_model_folder)

d2v = doc2vec.Doc2Vec.restore(doc2vec_model_folder)

print("d2v model loaded :-) ")

"""
Evaluate
"""

with d2v.sess.as_default():
    # print(d2v.doc_embeddings.eval().shape)
    # print(type(d2v.doc_embeddings.eval()))
    # pdistances = scipy.spatial.distance.pdist(d2v.doc_embeddings.eval(), metric=distance_measure)
    # distances = scipy.spatial.distance.squareform(pdistances)
    neigh = NearestNeighbors(n_neighbors=10, algorithm='brute', metric=distance_measure)
    neigh.fit(d2v.doc_embeddings.eval())
    sorted_distances_indices = []
    for embedding in d2v.doc_embeddings.eval():
        sorted_distances_indices.append(neigh.kneighbors(embedding.reshape(1, -1), 10, return_distance=False))

results = []
hits = []
# sorted_d2v_distance_indices = []
for doc_index, sorted_distance_indices in enumerate(sorted_distances_indices):
    # sorted_distance_indices = sorted(range(len(distance)), key=lambda x: distance[x], reverse=False)
    relevance_set = set(sorted_bm25_indices[doc_index][:10])
    hit = np.array([ix in relevance_set for ix in sorted_distance_indices[0][:10]], dtype=int)
    average_precision = rank_metrics.average_precision(hit)
    ndcg_at_10 = rank_metrics.ndcg_at_k(hit, 10)
    # sorted_d2v_distance_indices.append(sorted_distance_indices)
    hits.append(hit)
    results.append({
        'average_precision': average_precision,
        'ndcg_at_10': ndcg_at_10,
    })

with open('results_' + raw_data_folder + '_' + architecture + '.p', 'wb') as f:
    pickle.dump(results, f)

print("MAP: ", rank_metrics.mean_average_precision(hits))
print("MRR: ", rank_metrics.mean_reciprocal_rank(hits))
