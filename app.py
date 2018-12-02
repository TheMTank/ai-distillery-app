import datetime
import pickle
import os
import os.path

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from flask import Flask, request, send_from_directory, jsonify, render_template

from explorer import Model
from scripts.download_from_s3_bucket import download_file_from_s3

default_n = 15
STATIC_DIR = os.path.dirname(os.path.realpath(__file__)) + '/public'
CACHE = {}

app = Flask(__name__, static_folder='public', static_url_path='', template_folder="public/html")

# --------------------
# Routes to all HTML pages
# --------------------

@app.route("/")
def root():
    # return render_template('index.html')
    return send_from_directory('public/html', 'index.html')

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

# --------------------
# All other routes
# --------------------

@app.route("/word-embedding-proximity")
def get_word_embedding_proximity():
    # query params
    n = int(request.args.get('n', default_n))
    input_str = request.args.get('input_str')
    selected_word_embedding = request.args.get('type')

    input_str = input_str.lower()

    # inputted_word = inputted_word.replace(' ', '_') # todo probably keep
    # fuzzy match similar ones!!!!! and show

    print('Inputted string: {}. Embedding type: {}'.format(input_str, selected_word_embedding))

    if selected_word_embedding == 'gensim':
        gensim_labels_lowercase_strip = [x.lower().strip() for x in gensim_labels]
        if input_str in gensim_labels_lowercase_strip:
            print('Words most similar to:', input_str)
            similar_words, distances, sorted_idx = get_closest_vectors(gensim_labels, gensim_embeddings,
                                                           gensim_label_to_embeddings[input_str], n=n)
            response = [{'label': word, 'distance': round(float(dist), 5)} for word, dist in
                        zip(similar_words, distances)]
            print(response)
        else:
            response = ['Word not found']
    elif selected_word_embedding == 'fasttext':
        fasttext_labels_lowercase_strip = [x.lower().strip() for x in fasttext_labels]
        if input_str in fasttext_labels_lowercase_strip:
            print('Words most similar to:', input_str)
            similar_words, distances, sorted_idx = get_closest_vectors(fasttext_labels, fasttext_embeddings,
                                                           fasttext_label_to_embeddings[input_str], n=n)
            response = [{'label': word, 'distance': round(float(dist), 5)} for word, dist in
                        zip(similar_words, distances)]
            print(response)
        else:
            response = 'Word not found'
    else:
        response = 'Selected wrong word embedding'

    return jsonify(response)

@app.route("/paper-embedding-proximity")
def get_paper_embedding_proximity():
    print('Within paper embedding proximity table get ')
    # query params
    n = int(request.args.get('n', default_n))
    input_str = request.args.get('input_str')
    selected_embedding = request.args.get('type')

    input_str_clean = input_str.lower().strip()

    print('Inputted string: {}.\nInputted string clean: {}. Embedding type: {}'.format(input_str, input_str_clean, selected_embedding))

    if selected_embedding == 'lsa':
        lsa_labels_lowercase = [x.lower().strip() for x in lsa_labels]
        if input_str_clean in lsa_labels_lowercase:
            print('Labels most similar to:', input_str)
            similar_papers, distances, sorted_indices = get_closest_vectors(lsa_labels, lsa_embeddings,
                                                           lsa_label_to_embeddings[input_str], n=n)
            response = [{'label': label, 'distance': round(float(dist), 5),
                         'id': lsa_ids[sorted_idx], 'abstract': lsa_abstracts[sorted_idx]
                         } for label, dist, sorted_idx in zip(similar_papers, distances, sorted_indices)]
            print(response)
        else:
            response = ['Paper not found']
    elif selected_embedding == 'doc2vec':
        doc2vec_labels_lowercase = [x.lower().strip() for x in doc2vec_labels]
        if input_str_clean in doc2vec_labels_lowercase:
            print('Labels most similar to:', input_str)
            similar_words, distances, sorted_idx = get_closest_vectors(doc2vec_labels, doc2vec_embeddings,
                                                           doc2vec_label_to_embeddings[input_str], n=n)
            response = [{'label': word, 'distance': round(float(dist), 5)} for word, dist in
                        zip(similar_words, distances)]
            print(response)
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
    selected_embedding = request.args.get('type', 'tfidf')

    if selected_embedding == 'tfidf':
        # features = doc_tfidf_features
        model = tfidf_IR_model
    elif selected_embedding == 'lsa':
        model = lsa_IR_model
        # features = doc_lsa_features
    else:
        labels = ['embedding_type not found']

    preprocessed_query = query.lower()
    print('Searching for query: {}'.format(preprocessed_query))
    query_feats = model['model'].transform([preprocessed_query])#.toarray()

    closest_papers_titles, distances, sorted_indices = get_closest_vectors(model['titles'],
                                                                           model['feats'],
                                                                           query_feats, n=100,
                                                                           sparse=True)

    print('Closest paper titles top 5: {}'.format(closest_papers_titles[0:5]))
    top_paper_ids = np.array(model['ids'])[sorted_indices]
    top_paper_abstracts = np.array(model['abstracts'])[sorted_indices]

    response_obj = [{'title': title, 'paper_id': paper_id, 'abstract': abstract, 'distance': round(distance, 4)} for title,
                        paper_id, abstract, distance in zip(closest_papers_titles, top_paper_ids, top_paper_abstracts, distances)]

    return jsonify(response_obj)

@app.route("/api/explore")
def explore():
    query = request.args.get('query', '')
    limit = request.args.get('limit', '1000')
    enable_clustering = 'True'
    num_clusters = request.args.get('num_clusters', '30')
    embedding_type = request.args.get('embedding_type', 'gensim')

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

    print('Embedding type: {}. embedding_model: {}'.format(embedding_type, embedding_model))

    cache_key = '-'.join([query, limit, enable_clustering, num_clusters, embedding_type])
    result = CACHE.get(cache_key, {})
    if len(result) > 0:
        return jsonify({'result': CACHE[cache_key], 'cached': True})
    try:
        exploration = embedding_model.explore(query, limit=int(limit))
        exploration.reduce()
        if len(enable_clustering):
            if (len(num_clusters)):
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
    print(limit)
    print(queries)

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

    print('Embedding type: {}. embedding_model: {}'.format(embedding_type, embedding_model))

    try:
        # for i in range(len(queries)):
        #     queries[i] = queries[i].strip().lower()
        #       if embedding_model is word_embedding_model:
        #       queries[i] = queries[i].replace(' ', '_')
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

def get_model_obj(model_object_path):
    print('Loading embeddings at path: {}'.format(model_object_path))
    with open(model_object_path, 'rb') as handle:
        model_obj = pickle.load(handle, encoding='latin1')
        # labels = model_obj['labels']
        # embeddings = model_obj['embeddings']
        # label_to_embeddings = {label: embeddings[idx] for idx, label in
        #                        enumerate(labels)}
        print('Num ids: {}'.format(len(model_obj['ids'])))
        print('Num titles: {}'.format(len(model_obj['titles'])))
        print('feats shape: {}'.format(model_obj['feats'].shape))
        print('Model: {}'.format(model_obj['model']))
        if model_obj.get('abstracts'):
            print('Abstract shape: {}'.format(len(model_obj['abstracts'])))
        model_obj['model'].input = 'content'
        #model_obj['feats'] = model_obj['feats'].toarray()
        #model_obj['model'].named_steps.tfidf_vectorizer.input = 'content'

        return model_obj

def get_embedding_objs(embedding_path):
    print('Loading embeddings at path: {}'.format(embedding_path))
    with open(embedding_path, 'rb') as handle:
        embedding_obj = pickle.load(handle, encoding='latin1')
        labels = embedding_obj['labels']
        embeddings = embedding_obj['embeddings']
        label_to_embeddings = {label: embeddings[idx] for idx, label in
                                      enumerate(labels)}
        print('Num vectors: {}'.format(len(labels)))

        return labels, embeddings, label_to_embeddings

def download_model(key, output_path):
    """
    Function downloads necessary files from S3 bucket on server startup
    :param key: path within S3 bucket to get model
    :param output_path: where to store the downloaded model

    """

    if os.path.exists(output_path):
        print('File at: {} already exists'.format(output_path))
    else:
        download_file_from_s3(key, output_path)

# All models and saved objects
# ------------------
# gensim word2vec word embeddings (2d + 100d)
# fastText word embeddings
# LSA paper paper embeddings (2d + 100d + 300d)
# doc2vec paper embeddings (2d + 100d)
# ------------------

# Download all models if they don't already exist (download_model() checks)
gensim_embedding_name = 'type_word2vec#dim_100#dataset_ArxivNov4#time_2018-11-13T07_17_46.600182'
gensim_2d_embeddings_name = 'type_word2vec#dim_2#dataset_ArxivNov4#time_2018-11-13T07_17_46.600182'
# fasttext_embedding_name = 'type_fasttext#dim_300#dataset_ArxivNov4th#time_2018-11-23T04_35_16.470276'
fasttext_embedding_name = 'type_fasttext#dim_100#dataset_ArxivNov4th#time_2018-11-22T01_00_16.104601'
fasttext_2d_embedding_name = 'type_fasttext#dim_2#dataset_ArxivNov4th#time_2018-11-22T01_00_16.104601'
lsa_embedding_name = 'lsa-100.pkl' # 'lsa-300.pkl' # seems too big
lsa_embedding_2d_name = 'lsa-2.pkl'
lsa_info_object_name = 'LSA_info_object_54797.pkl'
lsa_IR_model_object_name = 'lsa-tfidf-pipeline-50k-feats-400-dim.pkl'
tfidf_IR_model_object_name = 'tfidf-50k-feats-IR-object.pkl' #'tfidf-10k-feats-IR-object.pkl' #'tfidf-25k-feats-IR-object.pkl' #'tfidf-50k-feats-IR-object.pkl'  #'tfidf-200k-feats-IR-object.pkl'
doc2vec_embedding_name = 'type_doc2vec#dim_100#dataset_ArxivNov4#time_2018-11-14T02_10_25.587584' # 'doc2vec-300.pkl' # not right format
doc2vec_embedding_2d_name = 'type_doc2vec#dim_2#dataset_ArxivNov4#time_2018-11-14T02_10_25.587584'

gensim_embedding_path = 'data/word_embeddings/' + gensim_embedding_name
gensim_2d_embeddings_path = 'data/word_embeddings/' + gensim_2d_embeddings_name
fasttext_embedding_path = 'data/word_embeddings/' + fasttext_embedding_name
fasttext_2d_embedding_path = 'data/word_embeddings/' + fasttext_2d_embedding_name
lsa_embedding_path = 'data/paper_embeddings/' + lsa_embedding_name
lsa_embedding_2d_path = 'data/paper_embeddings/' + lsa_embedding_2d_name
lsa_info_object_path = 'data/paper_embeddings/' + lsa_info_object_name
lsa_IR_model_object_path = 'data/models/' + lsa_IR_model_object_name
tfidf_IR_model_object_path = 'data/models/' + tfidf_IR_model_object_name
doc2vec_embedding_path = 'data/paper_embeddings/' + doc2vec_embedding_name
doc2vec_embedding_2d_path = 'data/paper_embeddings/' + doc2vec_embedding_2d_name

print('Beginning to download all models')
MODEL_OBJECTS_S3_PATH = 'model_objects/'
download_model(MODEL_OBJECTS_S3_PATH + gensim_embedding_name, gensim_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + gensim_2d_embeddings_name, gensim_2d_embeddings_path)
download_model(MODEL_OBJECTS_S3_PATH + lsa_embedding_name, lsa_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + lsa_embedding_2d_name, lsa_embedding_2d_path)
download_model(MODEL_OBJECTS_S3_PATH + fasttext_embedding_name, fasttext_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + fasttext_2d_embedding_name, fasttext_2d_embedding_path)
if not os.environ.get('IS_HEROKU'):
    download_model(MODEL_OBJECTS_S3_PATH + lsa_IR_model_object_name, lsa_IR_model_object_path)
download_model(MODEL_OBJECTS_S3_PATH + lsa_info_object_name, lsa_info_object_path)
download_model(MODEL_OBJECTS_S3_PATH + doc2vec_embedding_name, doc2vec_embedding_path)
download_model(MODEL_OBJECTS_S3_PATH + doc2vec_embedding_2d_name, doc2vec_embedding_2d_path)

# Loading models into embedding objects and Explorer objects
# Load all word embeddings
gensim_labels, gensim_embeddings, gensim_label_to_embeddings = get_embedding_objs(gensim_embedding_path)

# Load gensim 2d embedding model into word2vec-explorer visualisation
gensim_embedding_model = Model(gensim_2d_embeddings_path)

# fast_text_embedding_path = 'data/word_embeddings/fast_text_vectors.pkl'
fasttext_labels, fasttext_embeddings, fasttext_label_to_embeddings = get_embedding_objs(fasttext_embedding_path)

# Load fastText 2d embedding model into word2vec-explorer visualisation
fasttext_embedding_model = Model(fasttext_2d_embedding_path)

# Load paper embeddings
lsa_labels, lsa_embeddings, lsa_label_to_embeddings = get_embedding_objs(lsa_embedding_path)
doc2vec_labels, doc2vec_embeddings, doc2vec_label_to_embeddings = get_embedding_objs(doc2vec_embedding_path)
with open(lsa_info_object_path, 'rb') as f:
    lsa_info_object = pickle.load(f)
lsa_ids, lsa_abstracts = lsa_info_object['ids'], lsa_info_object['abstracts']

# Load lsa model (2d TSNE-precomputed) into word2vec-explorer visualisation
lsa_embedding_model = Model(lsa_embedding_2d_path)

# Load doc2vec model (2d TSNE-precomputed) into word2vec-explorer visualisation
doc2vec_embedding_model = Model(doc2vec_embedding_2d_path)

# Load IR model objects for Information Retrieval
if not os.environ.get('IS_HEROKU'):
    # lsa_IR_model = get_model_obj(lsa_IR_model_object_path)
    tfidf_IR_model = get_model_obj(tfidf_IR_model_object_path)

if __name__ == '__main__':
    print('Server has started up at time: {}'.format(datetime.datetime.now().
                                                     strftime("%I:%M%p on %B %d, %Y")))
    app.run(host='0.0.0.0', debug=True, use_reloader=True, port=80)  # 5000
    # app.run(host='0.0.0.0', debug=True, use_reloader=True, port=5000)  # 5000
