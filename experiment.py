import pickle
import os
import argparse

import doc2vec

import rank_metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # cancel optimization complaints by tensorflow


def get_relevance_labels():
    """
    Prepares or loads the relevance labels using bm25
    """

    if not os.path.isfile(rel_labels_fname):
        print('Computing relevance labels using bm25... (this takes a while, but should only be done once..)')
        import prepare_relevance_labels
        prepare_relevance_labels.prepare_relevance_labels(output_fname=rel_labels_fname, folder=raw_data_folder)
    with open(rel_labels_fname, 'rb') as rel_lab_file:
        _, _, _, tokenized, _, sorted_bm25_indices = pickle.load(rel_lab_file)
    return tokenized, sorted_bm25_indices


def train_paragraph_vectors():
    """
    Train paragraph vectors
    """

    print('Initializing and training paragraph vectors')
    d2v = doc2vec.Doc2Vec(**params)

    if params.architecture == 'pvdm':
        d2v.fit(tokenized)
    elif params.architecture == 'pvprior':
        d2v.fit(tokenized, sorted_bm25_indices)
    d2v.save(doc2vec_model_folder)
    return d2v


def train_paragraph_vectors_gensim():
    import gensim
    global tokenized
    tokenized = [gensim.models.doc2vec.TaggedDocument(toks, [i]) for i, toks in enumerate(tokenized)]
    model = gensim.models.Doc2Vec(tokenized,
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


def evaluate():
    """
    Evaluate
    """
    print("Evaluating embeddings agains BM25")
    neigh = NearestNeighbors(n_neighbors=10, algorithm='brute', metric=distance_measure)
    neigh.fit(embeddings)
    sorted_distances_indices = []
    for embedding in embeddings:
        sorted_distances_indices.append(neigh.kneighbors(embedding.reshape(1, -1), 10, return_distance=False))

    results = []
    hits = []
    for doc_index, sorted_distance_indices in enumerate(sorted_distances_indices):
        relevance_set = set(sorted_bm25_indices[doc_index][1:11])
        hit = np.array([ix in relevance_set for ix in sorted_distance_indices[0][:10]], dtype=int)
        average_precision = rank_metrics.average_precision(hit)
        ndcg_at_10 = rank_metrics.ndcg_at_k(hit, 10)
        hits.append(hit)
        results.append({
            'average_precision': average_precision,
            'ndcg_at_10': ndcg_at_10,
        })
        if doc_index % 1000 == 0:
            with open(os.path.join(exp_dir, 'results.p'), 'wb') as f:
                pickle.dump(results, f)

    with open(os.path.join(exp_dir, 'results.p'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('-algorithm',
                        type=str,
                        default='pvdm',
                        choices=['pvdm', 'pvprior', 'doc2vec-gensim'],
                        help='Sets the algorithm you want to use')
    parser.add_argument('-data',
                        type=str,
                        default='original_articles',
                        choices=['original_articles', 'aminer_org_v1', 'test'],
                        help='Name of data folder (placed in raw_data.tmp)')
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

    args = parser.parse_args()
    exp_name = '_'.join([str(val) for val in vars(args).values()])
    exp_dir = os.path.join('experiments', exp_name)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    print(args)
    raw_data_folder = args.data
    rel_labels_fname = 'relevance_labels_' + raw_data_folder + '.p'
    doc2vec_model_folder = os.path.join(exp_dir, 'model')
    architecture = args.algorithm

    distance_measure = args.dist_measure

    tokenized, sorted_bm25_indices = get_relevance_labels()

    embeddings = []
    params = {
        'batch_size': 128,
        'window_size': 8,
        'concat': True,
        'architecture': args.algorithm,
        'embedding_size_w': args.emb_size_w,
        'embedding_size_d': args.emb_size_d,
        'vocabulary_size': 50000,
        'document_size': len(tokenized),
        'loss_type': 'sampled_softmax_loss',
        'n_neg_samples': 64,
        'optimize': 'Adagrad',
        'learning_rate': 1.0,
        'n_steps': args.iterations * len(tokenized)
    }

    if args.algorithm == 'pvdm' or args.algorithm == 'pvprior':
        print("Training paragraph vectors")
        d2v = train_paragraph_vectors()
        print("Trained paragraph vectors")
        params = d2v.get_params()
        print("Loading embeddings")
        with d2v.sess.as_default():
            embeddings = d2v.doc_embeddings  # .eval()
    elif args.algorithm == 'doc2vec-gensim':
        print("Training paragraph vectors (gensim)")
        d2v = train_paragraph_vectors_gensim()
        print("Trained paragraph vectors")
        print("Loading embeddings")
        embeddings = d2v.docvecs

    evaluate()

    params.update(vars(args))
    with open(os.path.join(exp_dir, 'params.p'), 'wb') as params_file:
        pickle.dump(params, params_file)
