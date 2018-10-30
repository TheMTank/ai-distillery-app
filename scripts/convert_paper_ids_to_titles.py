import pickle

lsa_fp = 'data/paper_embeddings/lsa-300.pkl'
id_to_title_dict_fp = 'data/paper_id_to_title_dict.pkl'

with open(id_to_title_dict_fp, 'rb') as fp:
    paper_id_to_title_dict = pickle.load(fp)

print('Loading gensim vectors at path: {}'.format(lsa_fp))
with open(lsa_fp, 'rb') as handle:
    embedding_obj = pickle.load(handle, encoding='latin1')
    print('Num vectors: {}'.format(len(embedding_obj['labels'])))
    print('Shape of embeddings: {}'.format(embedding_obj['vectors'].shape))
    print('First 5: {}'.format(embedding_obj['labels'][0:5]))

    c = 0
    for idx, label in enumerate(embedding_obj['labels']):
        title = paper_id_to_title_dict.get(label)
        if title:
            c += 1
            embedding_obj['labels'][idx] = title

    print('{}/{} paper titles found'.format(c, len(embedding_obj['labels'])))
    if 'embeddings' not in embedding_obj:
        if 'vectors' in embedding_obj:
            embedding_obj['embeddings'] = embedding_obj.pop('vectors')

    new_path = 'data/paper_embeddings/lsa-300-converted.pkl'
    with open(new_path, 'wb') as f:
        pickle.dump(embedding_obj, f)
