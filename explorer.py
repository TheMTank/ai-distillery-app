import math
import gensim
import pickle
# import cPickle
import numpy as np
# from tsne import bh_sne
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine


class Exploration():

    def __init__(self, query, labels=[], vectors=[], already_2D=False):
        self.query = query
        self.parsed_query = {}
        self.labels = labels
        self.vectors = vectors
        self.reduction = []
        self.clusters = []
        self.distances = []
        self.stats = {}
        self.already_2D = already_2D

    def reduce(self):
        if not self.already_2D:
            print('Performing tSNE reduction ' +
                  'on {} vectors'.format(len(self.vectors)))
            self.reduction = TSNE(n_components=2, verbose=1).fit_transform(
                np.array(self.vectors, dtype=np.float64))  # slower than below
            # replaced below tsne with scikit's above
            # self.reduction = bh_sne(np.array(self.vectors, dtype=np.float64))
        else:
            print('Already 2D, no TSNE needed')
            self.reduction = np.array(self.vectors, dtype=np.float64)


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


class EmbeddingModel(object):
    def __init__(self, embeddings):
        self.vocab = {}


def get_closest_vectors(labels, all_vectors, vector_to_compare, n=5):
    # distances = np.array([np.linalg.norm(vec - vector_to_compare) for vec in all_vectors])
    distances = np.linalg.norm(all_vectors - vector_to_compare, axis=1) # vectorised
    sorted_idx = np.argsort(distances)  # [::-1]

    return list(zip(list(np.array(labels)[sorted_idx][0:n]), [x.item() for x in list(distances[sorted_idx][0:n])]))


class Model(object):
    """
    Instead of taking a gensim object designed very specifically for word embeddings (word2vec), we
    have created the BF format (Ben Format) which is a pickled dictionary like this:
    {
        'labels': ['label1', 'label2],
        'embeddings': np.array((vocab_size, embedding_size))
    }

    This format enables us to take embeddings calculated from many different packages
    (PyTorch, gensim, graphs, scikit-learn, etc) and therefore standardizes the expected input. The
    Model object then allows us to explore the space of embeddings and visualise them.
    """

    def __init__(self, filename):
        # try:
        # self.model1 = gensim.models.Word2Vec.load(filename)
        # self.model = EmbeddingModel(self.model1.wv.vocab)
        # todo allow option above to take gensim without conversion, otherwise below

        with open(filename, 'rb') as handle:
            print('Attempting to open file at: ', filename)
            embeddings_object = pickle.load(handle, encoding='latin1')
            self.vocab = embeddings_object['labels']
            self.embeddings_array = embeddings_object['embeddings']
            self.embeddings_dict = {self.vocab[i]: self.embeddings_array[i] for i in range(len(self.vocab))}
            print('Finished reading file and creating embeddings dictionary')

        self.already_2D = self.embeddings_array.shape[1] == 2

        # ideally the only input should be labels and embeddings in one file
        # how to convert from word2vec explorer into an embedding explorer
        # self.model.wv is a <gensim.models.keyedvectors.Word2VecKeyedVectors object at 0x7f594822b210>
        # self.model.wv.vocab is a dict full of of keyedVectors
        # self.model.wv.vocab['cat'] gets the specific keyedvector which holds the count and index
        # self.model['cat'] gets the numpy array

        # self.model.n_similarity() "Compute cosine similarity between two sets of words." returns float
        # self.model.most_similar() returns a list of tuples (word, score)
        '''except cPickle.UnpicklingError:
            load = gensim.models.Word2Vec.load_word2vec_format
            self.model = load(filename, binary=True)'''

    def compare(self, queries, limit):
        all_words = []
        comparison_words = []
        for query in queries:
            positive, negative = self._parse_query(query)
            comparison_words.append(positive[0])
            words, vectors, distances = self._most_similar_vectors(positive, negative, limit)
            all_words += words

        matrix = []
        labels = []
        for word in all_words:
            coordinates = []
            for word2 in comparison_words:
                # distance_model1 = self.model1.n_similarity([word2], [word])
                distance = 1 - cosine(self.embeddings_dict[word2], self.embeddings_dict[word])

                coordinates.append(distance)
            matrix.append(coordinates)
            labels.append(word)

        return {'labels': labels, 'comparison': matrix}

    def explore(self, query, limit=1000):
        print('Model#explore query={}, limit={}'.format(query, limit))
        exploration = Exploration(query, already_2D=self.already_2D)
        if len(query):
            print('Finding')
            positive, negative = self._parse_query(query)
            exploration.parsed_query['positive'] = positive
            exploration.parsed_query['negative'] = negative
            labels, vectors, distances = self._most_similar_vectors(positive, negative, limit)
            exploration.labels = labels
            exploration.vectors = vectors
            exploration.distances = distances
            print('first n labels and distances', labels[0:3], distances[0:3])
        else:
            print('Showing all vectors')
            exploration.labels, exploration.vectors, sample_rate = self._all_vectors(limit)
            exploration.stats['sample_rate'] = sample_rate
        # exploration.stats['vocab_size'] = len(self.model.wv.vocab)
        exploration.stats['vocab_size'] = len(self.vocab)
        exploration.stats['num_vectors'] = len(exploration.vectors)
        return exploration

    def _most_similar_vectors(self, positive, negative, limit):
        print('Model#_most_similar_vectors' +
              'positive={}, negative={}, limit={}'.format(positive, negative, limit))
        # results_from_model1 = self.model1.most_similar(positive=positive, negative=negative, topn=limit)
        results = get_closest_vectors(self.vocab, self.embeddings_array, self.embeddings_dict[positive[0]], n=limit)

        labels = []
        vectors = []
        distances = []
        for key, distance in results:
            # for key, distance in zip(results[0][1:], results[1][1:]):
            distances.append(distance)
            labels.append(key)
            vectors.append(self.embeddings_dict[key])
            # vectors.append(self.model[key])
        return labels, vectors, distances

    def _parse_query(self, query):
        expressions = query.split(' AND ')
        positive = []
        negative = []
        for expression in expressions:
            if expression.startswith('NOT '):
                negative.append(expression[4:])
            else:
                positive.append(expression)
        return positive, negative

    def _all_vectors(self, limit):
        sample = 1
        if limit > -1:
            # sample = int(math.ceil(len(self.model.wv.vocab) / limit))
            sample = int(math.ceil(len(self.vocab) / limit))
        # sample_rate = float(limit) / len(self.model.wv.vocab)
        sample_rate = float(limit) / len(self.vocab)
        print('Model#_most_similar_vectors' +
              'sample={}, sample_rate={}, limit={}'.format(sample, sample_rate, limit))
        labels = []
        vectors = []
        i = 0
        for word in self.vocab:
            if (i % sample) == 0:
                vectors.append(self.embeddings_dict[word])
                labels.append(word)
            i += 1
        return labels, vectors, sample_rate
