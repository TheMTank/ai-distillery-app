import os
import glob
import pickle
import logging
from elasticsearch_dsl.connections import connections
from models import elastic_model

# lets load the existing database to memory

def create_elastic_database(database_path, corpus_path):
    """
    Create an elastic serach database containing all the papers (with metadata) from an input directory
    :param database_path:
    :param corpus_path:
    :return:
    """
    try:
        db = pickle.load(open(database_path, 'rb'))
    except Exception as e:
        logging.error('error loading existing database:')
        logging.error(e)
        logging.error('starting from an empty database')
        db = {}

    def get_full_paper_id(full_paper_id):
        return full_paper_id.split('/')[-1]
    def get_paper_id_without_version(full_paper_id):
        return full_paper_id.split('/')[-1].split('v')[0]

    assert get_full_paper_id('http://arxiv.org/abs/1804.03131v1') == '1804.03131v1'
    assert get_paper_id_without_version('http://arxiv.org/abs/1804.03131v1') == '1804.03131'


    # Get full_paper_ids, titles, abstracts, full_text, authors, date
    all_papers_txt_file_paths = glob.glob(os.path.join(corpus_path, "*"))

    full_paper_id_to_title = {get_full_paper_id(doc['id']): doc['title'] for k, doc in db.items()}
    full_paper_id_to_abstract = {get_full_paper_id(doc['id']): doc['summary'] for k, doc in db.items()}
    full_paper_id_to_date = {get_full_paper_id(doc['id']): doc['published'] for k, doc in db.items()}
    full_paper_id_to_authors = {get_full_paper_id(doc['id']): [x['name'] for x in doc['authors']] for k, doc in db.items()}

    logging.info('{}, {}, {}, {}'.format(len(full_paper_id_to_title), len(full_paper_id_to_abstract),
                                  len(full_paper_id_to_date), len(full_paper_id_to_authors)))

    all_papers_full_paper_ids_and_paths = [(x, x.split('/')[-1].split('.pdf')[0]) for x in all_papers_txt_file_paths]
    # there are 6k+ txt files without an entry in the database, but there is ALWAYS another version
    # So only store the papers we have in the database
    all_papers_full_paper_ids_and_paths = [x for x in all_papers_full_paper_ids_and_paths if x[1] in full_paper_id_to_title]

    all_papers_full_text = []
    for fp, paper_id in all_papers_full_paper_ids_and_paths:
        with open(fp) as f:
            content = f.read()
            all_papers_full_text.append(content)

    all_papers_full_paper_ids = [x[1] for x in all_papers_full_paper_ids_and_paths]
    all_paper_titles = [full_paper_id_to_title[full_paper_id] for full_paper_id in all_papers_full_paper_ids]
    all_papers_abstracts = [full_paper_id_to_abstract[full_paper_id] for full_paper_id in all_papers_full_paper_ids]
    all_papers_authors = [full_paper_id_to_authors[full_paper_id] for full_paper_id in all_papers_full_paper_ids]
    all_papers_date = [full_paper_id_to_date[full_paper_id] for full_paper_id in all_papers_full_paper_ids]

    db_paper_ids_not_in_text_ids = [x for x in list(full_paper_id_to_title.keys()) if x not in all_papers_full_paper_ids]
    text_ids_not_in_db_paper_ids = [x for x in all_papers_full_paper_ids if x not in full_paper_id_to_title]

    logging.info('db_paper_ids_not_in_text_ids len: {}. text_ids_not_in_db_paper_ids len: {}'.format(len(db_paper_ids_not_in_text_ids),
                                                                                              len(text_ids_not_in_db_paper_ids)))
    connections.create_connection(hosts=['localhost'])


    for idx, (paper_id, title, abstract, full_text, authors, date) in \
            enumerate(zip(all_papers_full_paper_ids, all_paper_titles, all_papers_abstracts,
                        all_papers_full_text, all_papers_authors, all_papers_date)):

        if idx % 10 == 0:
            logging.info('Adding document {}/{}'.format(idx, len(all_paper_titles)))

        article = elastic_model.Paper(meta={'id': paper_id}, title=title)
        article.abstract = abstract
        article.authors = authors
        article.date = date
        article.full_text = full_text
        article.save()

def main():
    """
    Usage: python create_elastic_db.py --database /home/data/data_jan.p --corpus /home/data/text
    :return:
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help="databasae path (containing all data)")
    parser.add_argument("--corpus", help="text corpous path (containing text)")
    args = parser.parse_args()

    create_elastic_database(args.database, args.corpus)
