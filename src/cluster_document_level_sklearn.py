import logging
import os
import argparse
import sys
import json
from subprocess import check_output
from timeit import default_timer as timer
import resource

# allow import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sentencepiece as spm

from tokenizers.implementations import (
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
)
from functools import partial

from utils import save_script, get_output_path
from notebook_utils import parallel_list_map, load_document_lines


from notebook_utils import SpTokenizer
from notebook_utils import compute_counters
from collections import Counter
from notebook_utils import counter_to_freqs


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn import metrics

logging.basicConfig(level=logging.INFO)


def encode(tokenizer, doc):
    result = []
    for lines in doc:
        result += tokenizer.encode(lines, out_type=str)
    return result


from timeit import default_timer as timer

last_time = timer()


def usage_and_time():
    global last_time
    message = f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB, Time: {timer() - last_time:.2f} s"
    last_time = timer()
    return message


def main(args):
    output_path = get_output_path(
        args.output_dir,
        args.output_prefix,
        args={
            "k": args.k,
            "max_documents": args.max_documents,
            "token_freq_cutoff": args.token_freq_cutoff,
        },
        defaults={"k": 100, "max_documents": 0, "token_freq_cutoff": 2},
        always_include=["k"],
    )
    os.makedirs(output_path, exist_ok=True)

    logging.info(
        f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )
    logging.info(f"Loading data {usage_and_time()}")
    original_data = [
        load_document_lines(file, max_documents=args.max_documents)
        for file in args.input
    ]
    # flatten list of lists
    # original_data = one document per element
    original_input_file = [
        input_file
        for input_file, docs in zip(args.input, original_data)
        for doc in docs
    ]
    doc_line_numbers = []
    for docs in original_data:
        i = 1
        for doc in docs:
            doc_line_numbers.append(i)
            i += len(doc) + 1

    original_data = [doc for docs in original_data for doc in docs]

    merged_tokenizer = SpTokenizer(args.tokenizer)

    logging.info(f"Tokenizing data {usage_and_time()}")
    # tokenized_docs = one document per element. Each element is a list of tokens
    tokenized_docs = parallel_list_map(partial(encode, merged_tokenizer), original_data)

    logging.info(f"Running Tfidf vectorizer {usage_and_time()}")

    def dummy(doc):
        return doc

    tf = TfidfVectorizer(
        tokenizer=dummy, preprocessor=dummy, min_df=args.token_freq_cutoff
    )

    features = tf.fit_transform(tokenized_docs)

    logging.info(f"Reducing dimensionality with LSA {usage_and_time()}")
    lsa = make_pipeline(TruncatedSVD(n_components=300), Normalizer(copy=False))
    t0 = time()
    X_lsa = lsa.fit_transform(features)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    logging.info(f"LSA done in {time() - t0:.3f} s")
    logging.info(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

    logging.info(f"Clustering documents with k-means (k={args.k}) {usage_and_time()}")

    # cluster the data
    kmeans = MiniBatchKMeans(n_clusters=args.k, verbose=3, n_init=20).fit(X_lsa)

    logging.info(f"Clustering done {usage_and_time()}")
    logging.info(f"Saving the clustered documents to {output_path}")

    for i in range(args.k):
        with open(os.path.join(output_path, f"cluster_{i}.txt"), "w") as f:
            for j in np.argwhere(kmeans.labels_ == i)[:, 0]:
                for line in original_data[j]:
                    f.write(line + "\n")

    with open(os.path.join(output_path, f"doc_to_label.txt"), "w") as f:
        for src, linenumber, label in zip(
            original_input_file, doc_line_numbers, kmeans.labels_
        ):
            f.write(f"{src}:{linenumber}\t{label}\n")

    logging.info(f"Computing metrics {usage_and_time()}")

    with open(os.path.join(output_path, f"metrics.txt"), "w") as f:
        scores = {}
        labels = original_input_file
        scores["Homogeneity"] = metrics.homogeneity_score(labels, kmeans.labels_)
        scores["Completeness"] = metrics.completeness_score(labels, kmeans.labels_)
        scores["V-measure"] = metrics.v_measure_score(labels, kmeans.labels_)
        scores["Adjusted Rand-Index"] = metrics.adjusted_rand_score(
            labels, kmeans.labels_
        )
        scores["Silhouette Coefficient"] = metrics.silhouette_score(
            features, kmeans.labels_, sample_size=2000
        )
        json.dump(scores, f, indent=2)

    logging.info(f"Done computing metrics {usage_and_time()}")

    save_script(output_path)

    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_documents", type=int, default=0)
    parser.add_argument("-k", type=int, default=100)
    parser.add_argument("--token_freq_cutoff", type=int, default=2)

    args = parser.parse_args()
    main(args)
