import os
import time
import math
import pickle
import argparse

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class Exploration():

    def __init__(self, query, labels=[], vectors=[]):
        self.query = query
        self.parsed_query = {}
        self.labels = labels
        self.vectors = vectors
        self.reduction = []
        self.clusters = []
        self.distances = []
        self.stats = {}

    def reduce(self):
        print('Performing tSNE reduction ' +
              'on {} vectors'.format(len(self.vectors)))
        self.reduction = TSNE(n_components=2, verbose=1).fit_transform(
            np.array(self.vectors, dtype=np.float64))  # slower than below
        # replaced below tsne with scikit's above
        # self.reduction = bh_sne(np.array(self.vectors, dtype=np.float64))

    def cluster(self, num_clusters=30):
        clustering = KMeans(n_clusters=num_clusters)
        clustering.fit(self.reduction)
        self.clusters = clustering.labels_
        clustermatrix = []
        reduction = self.reduction.tolist()
        for cluster_id in range(num_clusters):
            clustermatrix.append([reduction[i]
                                  for i in range(len(self.vectors))
                                  if self.clusters[i] == cluster_id])
        self.cluster_centroids = clustering.cluster_centers_.tolist()
        self.cluster_centroids_closest_nodes = []
        for cluster_id in range(num_clusters):
            nodes_for_cluster = clustermatrix[cluster_id]
            centroid = self.cluster_centroids[cluster_id]
            closest_node_to_centroid = self._closest_node(
                centroid, nodes_for_cluster)
            coords = nodes_for_cluster[closest_node_to_centroid]
            node_id = reduction.index(coords)
            self.cluster_centroids_closest_nodes.append(node_id)

    def serialize(self):
        result = {
            'query': self.query,
            'parsed_query': self.parsed_query,
            'labels': self.labels,
            'stats': self.stats
        }
        if len(self.reduction) > 0:
            result['reduction'] = self.reduction.tolist()
        if len(self.distances) > 0:
            result['distances'] = self.distances
        if len(self.clusters) > 0:
            result['clusters'] = self.clusters.tolist()
            result['cluster_centroids'] = self.cluster_centroids
            closest_nodes = self.cluster_centroids_closest_nodes
            result['cluster_centroids_closest_nodes'] = closest_nodes
        return result

    def _closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2)

def _all_vectors(limit):
    sample = 1
    if limit > -1:
        # sample = int(math.ceil(len(self.model.wv.vocab) / limit))
        sample = int(math.ceil(len(vocab) / limit))
    # sample_rate = float(limit) / len(self.model.wv.vocab)
    sample_rate = float(limit) / len(vocab)
    print('Model#_most_similar_vectors' +
          'sample={}, sample_rate={}, limit={}'.format(sample, sample_rate, limit))
    labels = []
    vectors = []
    i = 0
    for word in vocab:
        # don't sample. Take all the vectors
        # if (i % sample) == 0:
        #     vectors.append(embeddings_dict[word])
        #     labels.append(word)
        vectors.append(embeddings_dict[word])
        labels.append(word)
        i += 1
    return labels, vectors, sample_rate


parser = argparse.ArgumentParser(description='Convert to 2d')
parser.add_argument('--input-path', type=str,
                    help='Input embedding path')
parser.add_argument('--output-path', type=str,
                    help='Output embedding')

args = parser.parse_args()

with open(args.input_path, 'rb') as handle:
    print('Attempting to open file at: ', args.input_path)
    embeddings_object = pickle.load(handle, encoding='latin1')
    vocab = embeddings_object['labels']
    embeddings_array = embeddings_object['embeddings']
    # self.embeddings_array = embeddings_object['vectors']
    embeddings_dict = {vocab[i]: embeddings_array[i] for i in range(len(vocab))}
    print('Finished reading file and creating embeddings dictionary')

    query = ''
    limit = 1000
    print('Model#explore query={}, limit={}'.format(query, limit))
    exploration = Exploration(query)
    print('Showing all vectors')
    exploration.labels, exploration.vectors, sample_rate = _all_vectors(limit)
    exploration.stats['sample_rate'] = sample_rate
    # exploration.stats['vocab_size'] = len(self.model.wv.vocab)
    exploration.stats['vocab_size'] = len(vocab)
    exploration.stats['num_vectors'] = len(exploration.vectors)

    print('Performing tSNE reduction ' +
          'on {} vectors'.format(len(exploration.vectors)))
    start = time.time()
    reduction = TSNE(n_components=2, verbose=1).fit_transform(
                     np.array(exploration.vectors, dtype=np.float64))

    print('Time taken for entire TSNE: {}'.format(time.time() - start))

    reduced_embeddings_obj_2d = {'labels': vocab, 'embeddings': reduction}

    with open(args.output_path, 'wb') as f:
        pickle.dump(reduced_embeddings_obj_2d, f)
        print('Saved 2d file to: {}'.format(args.output_path))


