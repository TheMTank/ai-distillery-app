import argparse
import glob

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser(description='Convert text files to tfidf features')
parser.add_argument('-i', '--input-folder', type=str,
                    help='Input folder path')
parser.add_argument('-o', '--tfidf-output-path', type=str,
                    help='Output TFIDF features')

args = parser.parse_args()

print('Input folder path: {}'.format(args.input_folder))
all_file_paths = glob.glob(args.input_folder)
print(all_file_paths[0:50])
print(len(all_file_paths))

# compute tfidf vectors with scikits
tfidf_vectorizer = TfidfVectorizer(input='filename', #input='content',
        encoding='utf-8', decode_error='replace', strip_accents='unicode',
        lowercase=True, analyzer='word', stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), #max_features = max_features,
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=1.0, min_df=1)

tf_idf_feats = tfidf_vectorizer.fit_transform(all_file_paths[0:1000])

print(tf_idf_feats.shape)


