import pickle
import os

import doc2vec
import scipy.spatial.distance

import rank_metrics
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # cancel optimization complaints by tensorflow

raw_data_folder = 'original_articles'
rel_labels_fname = 'relevance_labels_' + raw_data_folder + '.p'
doc2vec_model_folder = 'd2v.model/'

distance_measure = 'cosine'

"""
Get relevance labels
"""

if not os.path.isfile(rel_labels_fname):
    print('Computing relevance labels using bm25... (this takes a while, but should only be done once..)')
    import prepare_relevance_labels

    prepare_relevance_labels.prepare_relevance_labels(output_fname=rel_labels_fname, folder=raw_data_folder)
else:
    with open(rel_labels_fname, 'rb') as f:
        source_dict, docs, doc_names, tokenized, bm25_scores, sorted_bm25_indices = pickle.load(f)

"""
Train paragraph vectors
"""

# Always train, because restoring does not work (TODO: get restore function to work!)
if not False:  # os.path.exists(doc2vec_model_folder):
    print('Initializing and training paragraph vectors')
    d2v = doc2vec.Doc2Vec(batch_size=128,
                          window_size=8,
                          concat=True,
                          architecture='pvdm',
                          embedding_size_w=64,  # word embedding size
                          embedding_size_d=64,  # document embeding size
                          vocabulary_size=50000,
                          document_size=len(docs),
                          loss_type='sampled_softmax_loss',
                          n_neg_samples=64,
                          optimize='Adagrad',
                          learning_rate=1.0,
                          n_steps=100001  # 100001
                          )

    d2v.fit(tokenized)
    d2v.save(doc2vec_model_folder)
else:
    d2v = doc2vec.Doc2Vec.restore(doc2vec_model_folder)

"""
Evaluate
"""
distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(d2v.doc_embeddings, metric=distance_measure))

results = []
hits = []
for doc_index, distance in enumerate(distances):
    sorted_distance_indices = sorted(range(len(distance)), key=lambda x: distance[x], reverse=False)
    relevance_set = set(sorted_bm25_indices[doc_index][:10])
    hit = np.array([ix in relevance_set for ix in sorted_distance_indices[:10]], dtype=int)
    average_precision = rank_metrics.average_precision(hit)
    ndcg_at_10 = rank_metrics.ndcg_at_k(hit, 10)
    hits.append(hit)
    results.append({
        'average_precision': average_precision,
        'ndcg_at_10': ndcg_at_10,
    })

print("MAP: ", rank_metrics.mean_average_precision(hits))
print("MRR: ", rank_metrics.mean_reciprocal_rank(hits))

## MAP with   1000 steps: 0.022
## MAP with 100001 steps: 0.016
