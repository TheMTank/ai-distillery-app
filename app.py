import pickle
import sys
import os

import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify

from explorer import Model


default_n = 15
STATIC_DIR = os.path.dirname(os.path.realpath(__file__)) + '/public'
CACHE = {}

app = Flask(__name__, static_folder='public', static_url_path='')


def get_closest_vectors(labels, all_vectors, vector_to_compare, n=5):
    distances = np.linalg.norm(all_vectors - vector_to_compare, axis=1)  # vectorised
    sorted_idx = np.argsort(distances)

    return list(np.array(labels)[sorted_idx][0:n]), list(distances[sorted_idx][0:n])

@app.route("/")
def root():
    # return render_template('index.html')
    return send_from_directory('public/html', 'index.html')

@app.route("/charts")
def charts():
    return send_from_directory('public/html', 'all-charts.html')

@app.route("/word-embedding-table")
def word_embedding_table():
    return send_from_directory('public/html', 'word_embedding_table_similarity.html')


@app.route("/word-embedding-viz")
def word_embedding_viz():
    return send_from_directory('public/html', 'embedding_viz.html')

# could do it this way or cookie way (https://www.w3schools.com/js/js_cookies.asp)
# (https://stackoverflow.com/questions/13531149/check-for-a-cookie-with-python-flask)
# @app.route("/paper-embedding-viz")
# def paper_embedding_viz():
#     return send_from_directory('public/html', 'embedding_viz.html')

@app.route("/word_embedding_proximity")
def get_word_embedding():
    # query params
    n = int(request.args.get('n', default_n))
    inputted_word = request.args.get('word')
    selected_word_embedding = request.args.get('type')

    # inputted_word = inputted_word.strip().lower() # todo both upper and lower case atm. find lowercase version if not found
    # inputted_word = inputted_word.replace(' ', '_') # todo probably keep
    # fuzzy match similar ones!!!!! and show

    print('Inputted word: {}. Embedding type: {}'.format(inputted_word, selected_word_embedding))

    if selected_word_embedding == 'gensim':
        if inputted_word in gensim_labels:
            print('Words most similar to:', inputted_word)
            similar_words, distances = get_closest_vectors(gensim_labels, gensim_embeddings,
                                                           gensim_label_to_embeddings[inputted_word], n=15)
            response = [{'word': word, 'distance': round(float(dist), 5)} for word, dist in
                        zip(similar_words, distances)]
            print(response)
        else:
            response = 'Word not found'
    # elif selected_word_embedding == 'fast_text':
    #     if inputted_word in fast_text_labels:
    #         print('Words most similar to:', inputted_word)
    #         similar_words, distances = get_closest_vectors(fast_text_labels, fast_text_embeddings,
    #                                                        fast_text_label_to_embeddings[inputted_word], n=15)
    #         response = [{'word': word, 'distance': round(float(dist), 5)} for word, dist in
    #                     zip(similar_words, distances)]
    #         print(response)
    #     else:
    #         response = 'Word not found'
    else:
        response = 'Selected wrong word embedding'

    return jsonify(response)

@app.route("/get_embedding_labels")
def get_embedding_labels():
    selected_embedding = request.args.get('embedding_type', 'gensim')
    if selected_embedding == 'gensim':
        labels = gensim_labels

    return jsonify(labels)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('public/js', path)


@app.route('/styles/<path:path>')
def send_styles(path):
    return send_from_directory('public/styles', path)


# removed because quite slow.
@app.route("/api/explore")
def explore():
    query = request.args.get('query', '')
    limit = request.args.get('limit', '1000')
    enable_clustering = 'True'
    num_clusters = request.args.get('num_clusters', '30')
    embedding_type = request.args.get('embedding_type', 'gensim')

    if embedding_type == 'gensim':
        embedding_model = gensim_embedding_model
    elif embedding_type == 'lsa':
        embedding_model = lsa_embedding_model

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
    # queries = request.args.get('queries')
    # queries = queries.split(';')
    embedding_type = request.args.get('embedding_type', 'gensim')
    print(limit)
    print(queries)

    if embedding_type == 'gensim':
        embedding_model = gensim_embedding_model
    elif embedding_type == 'lsa':
        embedding_model = lsa_embedding_model
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


# All models and saved objects
# ------------------
# gensim word2vec embeddings
# fastText word embeddings
# LSA paper embeddings
# ------------------

# model = gensim.models.Word2Vec.load(args.word2vec_model_path)
# vocab = list(model.wv.index2word)

# Load all word embeddings
gensim_embedding_path = 'data/word_embeddings/gensim_vectors.pkl'
print('Loading gensim vectors at path: {}'.format(gensim_embedding_path))
with open(gensim_embedding_path, 'rb') as handle:
    gensim_embedding_obj = pickle.load(handle, encoding='latin1')
    gensim_labels = gensim_embedding_obj['labels']
    gensim_embeddings = gensim_embedding_obj['embeddings']
    gensim_label_to_embeddings = {label: gensim_embeddings[idx] for idx, label in enumerate(gensim_labels)}
    print('Num vectors: {}'.format(len(gensim_labels)))

# Load gensim model into word2vec-explorer visualisation
# gensim_embedding_model = Model(gensim_embedding_path)
# Load gensim 2d embedding model into word2vec-explorer visualisation
gensim_2d_embeddings_path = 'data/word_embeddings/gensim_vectors_2d.pkl'
gensim_embedding_model = Model(gensim_2d_embeddings_path)

# fast_text_embedding_path = 'data/word_embeddings/fast_text_vectors.pkl'
# print('Loading fast_text vectors at path: {}'.format(fast_text_embedding_path))
# with open(fast_text_embedding_path, 'rb') as handle:
#     fast_text_embedding_obj = pickle.load(handle, encoding='latin1')
#     fast_text_labels = fast_text_embedding_obj['labels']
#     fast_text_embeddings = fast_text_embedding_obj['embeddings']
#     fast_text_label_to_embeddings = {label: fast_text_embeddings[idx] for idx, label in enumerate(fast_text_labels)}
#     print('Num vectors: {}'.format(len(fast_text_labels)))

# Load paper embeddings
# lsa_embedding_path = 'data/paper_embeddings/lsa-300-converted.pkl'
lsa_embedding_path = 'data/paper_embeddings/lsa-300-converted-2d.pkl'
# Load lsa model into word2vec-explorer visualisation
lsa_embedding_model = Model(lsa_embedding_path)

if __name__ == '__main__':
    """
    """

    print('Listening')
    app.run(debug=True, use_reloader=True)
    # app.run(host='0.0.0.0', port=8080)
