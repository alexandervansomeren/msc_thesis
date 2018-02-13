import os
import argparse
import shutil
import time
import pickle

import nltk
import numpy as np
from sklearn.neighbors import NearestNeighbors

import utils


def train_paragraph_vectors():
    """
    Train paragraph vectors
    """

    print('Initializing and training paragraph vectors')
    d2v = doc2vec.Doc2Vec(**params)

    if params.architecture == 'pvdm':
        d2v.fit(tokenized_docs)
    elif params.architecture == 'pvprior':
        d2v.fit(tokenized_docs, links_per_doc(left_in_links_ixs))
    d2v.save(doc2vec_model_folder)
    return d2v


def train_gae():
    import networkx as nx
    import scipy.sparse as sp
    from external_scripts.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
    from gae.model import GCNModelAE, GCNModelVAE

    from gae.optimizer import OptimizerAE, OptimizerVAE
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score

    model_str = 'gcn_ae'

    # load data
    print("Loading data")
    nx_data = nx.from_dict_of_lists(utils.links_per_doc_creator(links, left_in_links_ixs))
    adj = nx.adjacency_matrix(nx_data)
    # features = np.array([])
    nodes = nx_data.nodes()
    del nx_data

    features = sp.identity(adj.shape[0])

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    print("Masking edges")
    adj_train, train_edges, val_edges, val_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_roc_score(edges_pos, edges_neg, emb=None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    # cost_val = []
    # acc_val = []
    val_roc_score = []

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    print("Starting training")
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
        val_roc_score.append(roc_curr)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    return nodes, emb


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('-algorithm',
                        type=str,
                        default='aai',
                        choices=['pvdm', 'pvprior', 'doc2vec-gensim', 'gae', 'gae-features', 'random', 'rai', 'jc',
                                 'aai', 'pa'],
                        help='Sets the algorithm you want to use')
    parser.add_argument('-data',
                        type=str,
                        default='dblp_v10',
                        choices=['aminer_org_v1', 'dblp_v10'],
                        help='Name of data folder (placed in data.tmp)')
    parser.add_argument('-sample',
                        type=int,
                        default=5000,
                        help='Sample from data set')
    parser.add_argument('-iterations',
                        type=int,
                        default=20,
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
    parser.add_argument('-sample_strategy',
                        type=str,
                        default='expand',
                        choices=['expand', 'binned'],
                        help='Sampling strategy)')
    parser.add_argument('-param_only',
                        type=bool,
                        default=False,
                        help='Only write the parameters (for debug purpose)')

    args = parser.parse_args()
    exp_name = '_'.join([str(val) for val in vars(args).values()][:-1])
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
    data_folder = args.data

    doc2vec_model_folder = os.path.join(exp_dir, 'model')
    architecture = args.algorithm

    n_folds = 5.

    distance_measure = args.dist_measure
    sample_seed = args.seed

    docs = []
    doc_names = []
    links = []

    print("Reading and sampling data...")
    docs, doc_names, links = utils.read_and_sample_data(data_folder, args.sample, docs, doc_names, links,
                                                        sample_seed, strategy=args.sample_strategy)
    print("Done.")
    tokenized_docs = []
    for doc in docs:
        tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
        tokenized_docs.append(tokens)
    del docs

    params = {
        'n_fold_splitting_strategy': 'random',
        'n_links': len(links),
    }
    params.update(vars(args))

    if args.algorithm in ['pvdm', 'pvprior', 'doc2vec-gensim']:
        params.update({
            'batch_size': 128,
            'window_size': 8,
            'concat': True,
            'architecture': args.algorithm,
            # 'embedding_size_w': args.emb_size_w,
            # 'embedding_size_d': args.emb_size_d,
            'vocabulary_size': 50000,
            'n_docs': len(tokenized_docs),
            'loss_type': 'sampled_softmax_loss',
            'n_neg_samples': 64,
            'optimize': 'Adagrad',
            'learning_rate': 1.0,
            'n_steps': args.iterations * len(tokenized_docs) * ((n_folds - 1) / n_folds),
        })

    with open(os.path.join(exp_dir, 'params.p'), 'wb') as params_file:
        pickle.dump(params, params_file)

    if args.param_only:
        print("Wrote parameter file, exiting.")
        exit(1)

    if args.algorithm == "doc2vec-gensim":
        import gensim

        tokenized_docs = [gensim.models.doc2vec.TaggedDocument(toks, [i]) for i, toks in enumerate(tokenized_docs)]

        print("Training paragraph vectors (gensim)")
        d2v = train_paragraph_vectors_gensim()
        print("Trained paragraph vectors")
    elif args.algorithm == "gae":
        import tensorflow as tf

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # cancel optimization complaints by tensorflow
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', args.iterations, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
        flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
        flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
        flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

        flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
        flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
        flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

    print('Starting', int(n_folds), '-fold runs')

    for fold_number, split in enumerate(
            utils.split_into_k_folds(links, int(n_folds), strategy='random', sample_seed=2017)):

        fold_folder = os.path.join(exp_dir, 'fold_' + str(fold_number))
        if not os.path.exists(fold_folder):
            os.mkdir(fold_folder)

        left_in_links_ixs = split[0]
        left_out_links_ixs = split[1]
        print(left_out_links_ixs)

        print('In fold ' + str(fold_number) + '. Saving results in ', fold_folder)
        print('Writing ground_truth file for trec_eval.')

        links_per_doc = utils.links_per_doc_creator(links, left_out_links_ixs)
        with open(os.path.join(fold_folder, 'trec_rel_file.tmp'), 'w') as trec_rel_f:
            for doc, temp_links in links_per_doc.items():
                for temp_link in temp_links:
                    trec_rel_f.write(' '.join([doc_names[doc], '0', doc_names[temp_link], '1']) + '\n')
        print('Done.')

        if args.algorithm == 'pvdm' or args.algorithm == 'pvprior':
            import doc2vec

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
            with open(os.path.join(fold_folder, 'trec_top_file.tmp'), 'w') as trec_top_f:
                for i, doc_name in enumerate(doc_names):
                    # most_similar_indices = [res[0] for res in ]
                    for s, msi in enumerate(d2v.docvecs.most_similar(i, topn=50)):
                        trec_top_f.write(
                            ' '.join([doc_name, '0', doc_names[msi[0]], str(msi[1]), '0', exp_name]) + '\n')
        elif args.algorithm == 'random':
            print('Generating random results.')
            doc_names_set = set(doc_names)
            # print("doc_names contains duplicates?", len(doc_names_set) < len(doc_names))
            with open(os.path.join(fold_folder, 'trec_top_file.tmp'), 'w') as trec_top_f:
                for i, doc_name in enumerate(doc_names):
                    temp_set = doc_names_set.difference({doc_name})
                    most_similar = np.random.choice(list(temp_set), 50, replace=False)
                    for s, doc in enumerate(most_similar):
                        trec_top_f.write(' '.join([doc_name, '0', doc, str(s), '0', exp_name]) + '\n')
            print('Done.')
        elif args.algorithm == "gae" or args.algorithm == "gae-features":
            nodes, embeddings = train_gae()
            nodes = list(nodes)
            print(nodes)
            neigh = NearestNeighbors(n_neighbors=50, algorithm='brute', metric=distance_measure)
            neigh.fit(embeddings)
            with open(os.path.join(fold_folder, 'trec_top_file.tmp'), 'w') as trec_top_f:
                for source, embedding in zip(nodes, embeddings):
                    distances, indexes = neigh.kneighbors(embedding.reshape(1, -1), 50, return_distance=True)
                    for ix, distance in zip(indexes.tolist()[0], distances.tolist()[0]):
                        # print(ix)
                        target = nodes[ix]
                        # print(target)
                        trec_top_f.write(
                            ' '.join([doc_names[source], '0', doc_names[target], str(distance), '0', exp_name]) + '\n')

        elif args.algorithm == "" or args.algorithm == "rai" or args.algorithm == 'aai' or args.algorithm == 'pa':
            import networkx as nx
            import heapq

            G = nx.from_dict_of_lists(utils.links_per_doc_creator(links, left_in_links_ixs))

            with open(os.path.join(fold_folder, 'trec_top_file.tmp'), 'w') as trec_top_f:
                print('Generating ' + args.algorithm + ' baseline.')

                n_nodes = len(G.nodes())
                for c, u in enumerate(G.nodes()):
                    if args.algorithm == 'jc':  # jaccard_coefficient
                        preds = nx.jaccard_coefficient(G, utils.nonedges(G, u))
                    elif args.algorithm == 'rai':  # resource allocation index
                        preds = nx.resource_allocation_index(G, utils.nonedges(G, u))
                    elif args.algorithm == 'aai':  # adamic/adar
                        preds = nx.adamic_adar_index(G, utils.nonedges(G, u))
                    elif args.algorithm == 'pa':  # preferential attachment
                        preds = nx.preferential_attachment(G, utils.nonedges(G, u))
                    most_similar = heapq.nlargest(50, preds, key=lambda x: x[2])
                    for source_ix, target_ix, s in most_similar:
                        trec_top_f.write(
                            ' '.join([doc_names[source_ix], '0', doc_names[target_ix], str(s), '0', exp_name]) + '\n')
                    if c % 100 == 0:
                        print(c, ' / ', n_nodes)

        print('--------------------')
