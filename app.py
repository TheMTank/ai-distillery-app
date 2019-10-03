import datetime
import bz2
import pickle
import logging
import os
import os.path

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from flask import Flask, request, send_from_directory, jsonify, render_template
# from flask_sslify import SSLify

from explorer import Model
from scripts.download_from_s3_bucket import download_file_from_s3

from utilities import search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

default_n = 15
STATIC_DIR = os.path.dirname(os.path.realpath(__file__)) + '/public'
CACHE = {}

app = Flask(__name__, static_folder='public', static_url_path='', template_folder="public/html")
# sslify = SSLify(app) # if prod

# --------------------
# Routes to all HTML pages
# --------------------

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/i2")
def i2():
    return render_template('index2.html')

@app.route("/charts")
def charts():
    return render_template('charts.html')

@app.route("/paper-search-page")
def paper_search_page():
    return render_template('paper_search.html')

@app.route("/word-embedding-proximity-page")
def word_embedding_table():
    return render_template('word_embedding_proximity.html')

@app.route("/paper-embedding-proximity-page")
def paper_embedding_table():
    return render_template('paper_embedding_proximity.html')

@app.route("/word-embedding-viz")
def word_embedding_viz():
    return send_from_directory('public/html', 'word_embedding_viz.html')

@app.route("/paper-embedding-viz")
def paper_embedding_viz():
    return render_template('paper_embedding_viz.html')

# --------------------
# Routes to other files
# --------------------

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('public/js', path)

@app.route('/styles/<path:path>')
def send_styles(path):
    return send_from_directory('public/styles', path)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'public/images'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --------------------
# All other routes
# --------------------

@app.route("/word-embedding-proximity")
def get_word_embedding_proximity():
    # query params
    n = int(request.args.get('n', default_n))
    input_str = request.args.get('input_str')
    selected_word_embedding = request.args.get('type')

    input_str = input_str.lower().strip()

    # fuzzy match similar ones!!!!! and show todo?

    logger.info('Inputted string: {}. Embedding type: {}'.format(input_str, selected_word_embedding))

    if selected_word_embedding == 'gensim':
        if input_str in gensim_labels_lowercase:
            logger.info('Words most similar to: {}'.format(input_str))
            similar_words, distances, sorted_idx = get_closest_vectors(gensim_labels, gensim_embeddings,
                                                           gensim_label_to_embeddings[input_str], n=n)
            response = [{'label': word, 'distance': round(float(dist), 5)} for word, dist in
                        zip(similar_words, distances)]
            logger.info(response)
        else:
            response = ['Word not found']
    elif selected_word_embedding == 'fasttext':
        if input_str in fasttext_labels_lowercase:
            logger.info('Words most similar to: {}'.format(input_str))
            similar_words, distances, sorted_idx = get_closest_vectors(fasttext_labels, fasttext_embeddings,
                                                           fasttext_label_to_embeddings[input_str], n=n)
            response = [{'label': word, 'distance': round(float(dist), 5)} for word, dist in
                        zip(similar_words, distances)]
            logger.info(response)
        else:
            response = 'Word not found'
    else:
        response = 'Selected wrong word embedding'

    return jsonify(response)

@app.route("/paper-embedding-proximity")
def get_paper_embedding_proximity():
    # query params
    n = int(request.args.get('n', default_n))
    input_str = request.args.get('input_str')
    selected_embedding = request.args.get('type')

    # make sure lower and only 1 whitespace between words
    input_str_clean = ' '.join(input_str.strip().lower().split())

    logger.info('Inputted string: {}.\nInputted string clean: {}. Embedding type: {}'.format(input_str, input_str_clean, selected_embedding))

    if selected_embedding == 'lsa':
        if input_str_clean in lsa_labels_lowercase:
            logger.info('Labels most similar to: {}'.format(input_str))
            similar_papers, distances, sorted_indices = get_closest_vectors(lsa_labels, lsa_embeddings,
                                                           lsa_label_to_embeddings[input_str_clean], n=n)
            response = [{'label': label, 'distance': round(float(dist), 5),
                         'id': lsa_ids[sorted_idx], 'abstract': lsa_abstracts[sorted_idx]
                         } for label, dist, sorted_idx in zip(similar_papers, distances, sorted_indices)]
            logger.info(response)
        else:
            response = ['Paper not found']
    elif selected_embedding == 'doc2vec':
        if input_str_clean in doc2vec_labels_lowercase:
            logger.info('Labels most similar to: {}'.format(input_str))
            similar_words, distances, sorted_idx = get_closest_vectors(doc2vec_labels, doc2vec_embeddings,
                                                           doc2vec_label_to_embeddings[input_str_clean], n=n)
            response = [{'label': label, 'distance': round(float(dist), 5)} for label, dist in
                        zip(similar_words, distances)]
            logger.info(response)
        else:
            response = ['Paper not found']
    else:
        response = 'Selected wrong embedding'

    return jsonify(response)

@app.route("/get-embedding-labels")
def get_embedding_labels():
    selected_embedding = request.args.get('embedding_type', 'gensim')
    if selected_embedding == 'gensim':
        labels = gensim_labels
    elif selected_embedding == 'lsa':
        labels = lsa_labels
    elif selected_embedding == 'doc2vec':
        labels = doc2vec_labels
    else:
        labels = ['embedding_type not found']

    return jsonify(labels)

@app.route("/search-papers")
def search_papers():
    query = request.args.get('query', '')
    num_results = request.args.get('num_results', 10)
    from_ = request.args.get('from_', 0)
    twitter_popularity = request.args.get('defaultCheck1', False)
    data = search.elastic_search_papers(query, num_results, twitter_popularity, from_result=from_)

    return jsonify(data)

@app.route("/api/explore")
def explore():
    query = request.args.get('query', '')
    limit = request.args.get('limit', '1000')
    limit = str(min(int(5000), int(limit)))
    enable_clustering = 'True'
    num_clusters = request.args.get('num_clusters', '30')
    embedding_type = request.args.get('embedding_type', 'gensim')

    query = query.strip().lower()

    if embedding_type == 'gensim':
        embedding_model = gensim_embedding_model
    elif embedding_type == 'fasttext':
        embedding_model = fasttext_embedding_model
    elif embedding_type == 'lsa':
        embedding_model = lsa_embedding_model
    elif embedding_type == 'doc2vec':
        embedding_model = doc2vec_embedding_model
    else:
        embedding_model = gensim_embedding_model  # default model so it doesn't crash

    logger.info('Query: {}. Embedding type: {}. embedding_model: {}'.format(query, embedding_type, embedding_model))

    cache_key = '-'.join([query, limit, enable_clustering, num_clusters, embedding_type])
    result = CACHE.get(cache_key, {})
    if len(result) > 0:
        return jsonify({'result': CACHE[cache_key], 'cached': True})
    try:
        exploration = embedding_model.explore(query, limit=int(limit))
        exploration.reduce()
        if len(enable_clustering):
            if len(num_clusters):
                num_clusters = int(num_clusters)
            exploration.cluster(num_clusters=num_clusters)
        result = exploration.serialize()
        CACHE[cache_key] = result
        return jsonify({'result': result, 'cached': False})
    except KeyError:
        return jsonify({'error': {'message': 'No vector found for ' + query}})

@app.route("/api/compare")
def compare():
    limit = request.args.get('limit', 100)
    queries = request.args.getlist('queries[]')
    # queries = request.args.get('queries') # todo double check all browsers
    # queries = queries.split(';')
    embedding_type = request.args.get('embedding_type', 'gensim')
    logger.info(limit)
    logger.info(queries)

    if embedding_type == 'gensim':
        embedding_model = gensim_embedding_model
    elif embedding_type == 'fasttext':
        embedding_model = fasttext_embedding_model
    elif embedding_type == 'lsa':
        embedding_model = lsa_embedding_model
    elif embedding_type == 'doc2vec':
        embedding_model = doc2vec_embedding_model
    else:
        embedding_model = gensim_embedding_model  # default model so it doesn't crash

    logger.info('Embedding type: {}. embedding_model: {}'.format(embedding_type, embedding_model))

    try:
        queries = [query.strip().lower() for query in queries]
        result = embedding_model.compare(queries, limit=int(limit))
        return jsonify({'result': result})
    except KeyError:
        return jsonify({'error':
                            {'message': 'No vector found for {}'.format(queries)}})

# --------------------
# Helper functions
# --------------------

def get_closest_vectors(labels, all_vectors, query_vector, n=5, sparse=False):
    if sparse:
        distances = euclidean_distances(all_vectors, query_vector).flatten()
    else:
        distances = np.linalg.norm(all_vectors - query_vector, axis=1)  # vectorised # todo try scikit and many more and put in notebook
    sorted_idx = np.argsort(distances)

    return list(np.array(labels)[sorted_idx][0:n]), list(distances[sorted_idx][0:n]), sorted_idx[0:n]

def get_embedding_objs(embedding_path):
    logger.info('Loading embeddings at path: {}'.format(embedding_path))
    with open(embedding_path, 'rb') as handle:
        embedding_obj = pickle.load(handle, encoding='latin1')
        labels = embedding_obj['labels']
        labels = [' '.join(x.strip().split()) for x in labels]
        labels_lowercase = [x.lower().strip() for x in labels]
        embeddings = embedding_obj['embeddings']
        label_to_embeddings = {label: embeddings[idx] for idx, label in
                                      enumerate(labels_lowercase)}
        logger.info('Num vectors: {}'.format(len(labels)))

        return labels, labels_lowercase, embeddings, label_to_embeddings

def download_model(key, output_path):
    """
    Function downloads necessary files from S3 bucket on server startup
    :param key: path within S3 bucket to get model
    :param output_path: where to store the downloaded model

    """

    if os.path.exists(output_path):
        logger.info('File at: {} already exists'.format(output_path))
    else:
        download_file_from_s3(key, output_path)

# All models and saved objects
# ------------------
# gensim word2vec word embeddings (2d + 100d)
# fastText word embeddings
# LSA paper paper embeddings (2d + 100d + 300d)
# doc2vec paper embeddings (2d + 100d)
# Whoosh paper index
# ------------------

# Download all models if they don't already exist (download_model() checks)
gensim_embedding_name = 'type_word2vec#dim_100#dataset_ArxivNov4#time_2018-11-13T07_17_46.600182'
gensim_2d_embeddings_name = 'type_word2vec#dim_2#dataset_ArxivNov4#time_2018-11-13T07_17_46.600182'
fasttext_embedding_name = 'type_fasttext#dim_100#dataset_ArxivNov4th#time_2018-11-22T01_00_16.104601' #'type_fasttext#dim_300#dataset_ArxivNov4th#time_2018-11-23T04_35_16.470276'
fasttext_2d_embedding_name = 'type_fasttext#dim_2#dataset_ArxivNov4th#time_2018-11-22T01_00_16.104601'
lsa_embedding_name = 'lsa-100.pkl' # 'lsa-300.pkl' # seems too big
lsa_embedding_2d_name = 'lsa-2.pkl'
lsa_info_object_name = 'LSA_info_object_54797.pkl'
doc2vec_embedding_name = 'type_doc2vec#dim_100#dataset_ArxivNov4#time_2018-11-14T02_10_25.587584' # 'doc2vec-300.pkl' # not right format
doc2vec_embedding_2d_name = 'type_doc2vec#dim_2#dataset_ArxivNov4#time_2018-11-14T02_10_25.587584'

gensim_embedding_path = 'data/word_embeddings/' + gensim_embedding_name
gensim_2d_embeddings_path = 'data/word_embeddings/' + gensim_2d_embeddings_name
fasttext_embedding_path = 'data/word_embeddings/' + fasttext_embedding_name
fasttext_2d_embedding_path = 'data/word_embeddings/' + fasttext_2d_embedding_name
lsa_embedding_path = 'data/paper_embeddings/' + lsa_embedding_name
lsa_embedding_2d_path = 'data/paper_embeddings/' + lsa_embedding_2d_name
lsa_info_object_path = 'data/paper_embeddings/' + lsa_info_object_name
doc2vec_embedding_path = 'data/paper_embeddings/' + doc2vec_embedding_name
doc2vec_embedding_2d_path = 'data/paper_embeddings/' + doc2vec_embedding_2d_name

logger.info('Beginning to download all models')
MODEL_OBJECTS_S3_PATH = 'model_objects/'
download_model(MODEL_OBJECTS_S3_PATH + gensim_embedding_name, gensim_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + gensim_2d_embeddings_name, gensim_2d_embeddings_path)
download_model(MODEL_OBJECTS_S3_PATH + lsa_embedding_name, lsa_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + lsa_embedding_2d_name, lsa_embedding_2d_path)
download_model(MODEL_OBJECTS_S3_PATH + fasttext_embedding_name, fasttext_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + fasttext_2d_embedding_name, fasttext_2d_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + lsa_info_object_name, lsa_info_object_path)
download_model(MODEL_OBJECTS_S3_PATH + doc2vec_embedding_name, doc2vec_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + doc2vec_embedding_2d_name, doc2vec_embedding_2d_path)

# Loading models into embedding objects and Explorer objects
# Load all word embeddings
gensim_labels, gensim_labels_lowercase, gensim_embeddings, gensim_label_to_embeddings = get_embedding_objs(gensim_embedding_path)

# Load gensim 2d embedding model into word2vec-explorer visualisation
gensim_embedding_model = Model(gensim_2d_embeddings_path)

# fast_text_embedding_path = 'data/word_embeddings/fast_text_vectors.pkl'
fasttext_labels, fasttext_labels_lowercase, fasttext_embeddings, fasttext_label_to_embeddings = get_embedding_objs(fasttext_embedding_path)

# Load fastText 2d embedding model into word2vec-explorer visualisation
fasttext_embedding_model = Model(fasttext_2d_embedding_path)

# Load paper embeddings
lsa_labels, lsa_labels_lowercase, lsa_embeddings, lsa_label_to_embeddings = get_embedding_objs(lsa_embedding_path)
doc2vec_labels, doc2vec_labels_lowercase, doc2vec_embeddings, doc2vec_label_to_embeddings = get_embedding_objs(doc2vec_embedding_path)
with open(lsa_info_object_path, 'rb') as f:
    lsa_info_object = pickle.load(f)
lsa_ids, lsa_abstracts = lsa_info_object['ids'], lsa_info_object['abstracts']

# Load lsa model (2d TSNE-precomputed) into word2vec-explorer visualisation
lsa_embedding_model = Model(lsa_embedding_2d_path)

# Load doc2vec model (2d TSNE-precomputed) into word2vec-explorer visualisation
doc2vec_embedding_model = Model(doc2vec_embedding_2d_path)


if __name__ == '__main__':
    logger.info('Server has started up at time: {}'.format(datetime.datetime.now().
                                                     strftime("%I:%M%p on %B %d, %Y")))
    app.run(host='0.0.0.0', port=8080)
    # app.run(host='0.0.0.0', debug=True, use_reloader=True, port=5000)  # 5000
