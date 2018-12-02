import pickle


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

full_id_to_title_dict_fp = '../data/full_paper_id_to_title_dict.pkl'
id_to_title_dict_fp = '../data/paper_id_to_title_dict.pkl'

with open(full_id_to_title_dict_fp, 'rb') as fp:
    full_paper_id_to_title_dict = pickle.load(fp)
with open(id_to_title_dict_fp, 'rb') as fp:
    paper_id_to_title_dict = pickle.load(fp)

len(paper_id_to_title_dict)
title_to_paper_id_dict = {v:k for k, v in paper_id_to_title_dict.items()}
title_to_full_paper_id_dict = {v:k for k, v in full_paper_id_to_title_dict.items()}

lsa_embedding_name = 'lsa-100.pkl'
lsa_embedding_path = '../data/paper_embeddings/' + lsa_embedding_name
lsa_labels, lsa_embeddings, lsa_label_to_embeddings = get_embedding_objs(lsa_embedding_path)

lsa_ids = [title_to_full_paper_id_dict.get(label, 'Not found') for label in lsa_labels]

print(lsa_ids[0:100])
print('Num ids not found with label: {}'.format(len([x for x in lsa_ids if x != 'Not found'])))

# from utils import Config, safe_pickle_dump
# import stopwords

# lets load the existing database to memory
try:
    print()
    db = pickle.load(open('/home/beduffy/all_projects/arxiv-sanity-preserver/db.p', 'rb'))
except Exception as e:
    print('error loading existing database:')
    print(e)
    print('starting from an empty database')
    db = {}

def get_full_paper_id(full_paper_id):
    return full_paper_id.split('/')[-1]
def get_paper_id_without_version(full_paper_id):
    return full_paper_id.split('/')[-1].split('v')[0]
assert get_full_paper_id('http://arxiv.org/abs/1804.03131v1') == '1804.03131v1'
assert get_paper_id_without_version('http://arxiv.org/abs/1804.03131v1') == '1804.03131'

# import pdb;pdb.set_trace()
# lsa_abstracts = []

full_paper_id_to_abstract = {get_full_paper_id(doc['id']): doc['summary'] for k, doc in db.items()}


lsa_abstracts = [full_paper_id_to_abstract.get(full_paper_id, 'Not found') for full_paper_id in lsa_ids]
print(len(lsa_abstracts))
print(len(lsa_ids))

with open('../data/full_paper_id_to_abstract_dict.pkl', 'wb') as f:
    pickle.dump(full_paper_id_to_abstract, f)

path_to_LSA_info_object = '../data/paper_embeddings/LSA_info_object_{}.pkl'.format(len(lsa_abstracts))
with open(path_to_LSA_info_object, 'wb') as f:
    pickle.dump({
        'ids': lsa_ids,
        'abstracts': lsa_abstracts
    }, f)

