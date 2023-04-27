import logging
import os
import argparse
import sys
import json
from subprocess import check_output
from timeit import default_timer as timer
import resource
import pickle

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
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from scipy.sparse import csr_matrix, save_npz
from sklearn import metrics

import umap
from tqdm import tqdm

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


# taken from https://github.com/kernelmachine/cbtm/blob/main/metaseq/scripts/train_clusterer.py#L239
def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def load_data(input, max_documents):
    original_data = [
        load_document_lines(file, max_documents=max_documents) for file in input
    ]
    # flatten list of lists
    # original_data = one document per element
    original_input_file = [
        input_file for input_file, docs in zip(input, original_data) for doc in docs
    ]
    doc_line_numbers = []
    for docs in original_data:
        i = 1
        for doc in docs:
            doc_line_numbers.append(i)
            i += len(doc) + 1

    original_data = [doc for docs in original_data for doc in docs]

    return original_data, original_input_file, doc_line_numbers


def main(args):
    output_path = get_output_path(
        args.output_dir,
        args.output_prefix,
        args={
            "k": args.k,
            "vectorizer": args.vectorizer,
            "max_documents": args.max_documents,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "use_stop_words": args.use_stop_words,
            "ngram_range": args.ngram_range,
            "svd_components": args.svd_components,
        },
        defaults={
            "k": 100,
            "vectorizer": "sentencepiece",
            "max_documents": 0,
            "min_df": 2,
            "max_df": 1.0,
            "use_stop_words": False,
            "ngram_range": (1, 1),
            "svd_components": 100,
        },
        always_include=["k"],
    )
    os.makedirs(output_path, exist_ok=True)

    logging.info(f"Loading data")

    original_data, original_input_file, doc_line_numbers = load_data(
        args.input, args.max_documents
    )

    logging.info(f"Done loading data {usage_and_time()}")

    tfidf_args = {
        "min_df": args.min_df,
        "max_df": args.max_df,
        "ngram_range": args.ngram_range,
    }

    logging.info(f"Running vectorizer")

    if args.vectorizer == "sentencepiece":
        merged_tokenizer = SpTokenizer(args.tokenizer)

        logging.info(f"Tokenizing data")
        # tokenized_docs = one document per element. Each element is a list of tokens
        tokenized_docs = parallel_list_map(
            partial(encode, merged_tokenizer), original_data
        )
        logging.info(f"Done tokenizing data {usage_and_time()}")

        def dummy(doc):
            return doc

        vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, **tfidf_args)
        X_tfidf = vectorizer.fit_transform(tqdm(tokenized_docs))
    if args.vectorizer == "word":
        stop_words = None
        if args.use_stop_words:
            stop_words = list(ENGLISH_STOP_WORDS.union(["#NUMBER"]))

        vectorizer = NumberNormalizingVectorizer(stop_words=stop_words, **tfidf_args)
        flat_docs = [" ".join(doc) for doc in original_data]
        X_tfidf = vectorizer.fit_transform(tqdm(flat_docs))
    if args.vectorizer == "char":
        vectorizer = NumberNormalizingVectorizer(analyzer="char_wb", **tfidf_args)
        flat_docs = [" ".join(doc) for doc in original_data]
        X_tfidf = vectorizer.fit_transform(tqdm(flat_docs))

    logging.info(f"Done running vectorizer {usage_and_time()}")

    with open(os.path.join(output_path, f"tfidf_model.pkl"), "wb+") as f:
        _ = pickle.dump(vectorizer, f)

    save_npz(os.path.join(output_path, "tfidf_embeddings.npz"), X_tfidf)

    logging.info(f"Reducing dimensionality with LSA {usage_and_time()}")
    lsa = make_pipeline(
        TruncatedSVD(n_components=args.svd_components), Normalizer(copy=False)
    )
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    with open(os.path.join(output_path, f"lsa_model.pkl"), "wb+") as f:
        _ = pickle.dump(lsa, f)

    np.save(os.path.join(output_path, "lsa_embeddings.npy"), X_lsa)

    logging.info(f"LSA done in {usage_and_time()}")
    logging.info(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

    logging.info(f"Reducing dimensionality with UMAP {usage_and_time()}")
    reducer = umap.UMAP(verbose=True)
    X_umap = reducer.fit_transform(X_lsa)
    logging.info(f"UMAP done {usage_and_time()}")

    np.save(os.path.join(output_path, "umap_embeddings.npy"), X_umap)

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
                f.write("\n")

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
            X_tfidf, kmeans.labels_, sample_size=2000
        )
        scores["Silhouette Coefficient on embeddings"] = metrics.silhouette_score(
            X_lsa, kmeans.labels_, sample_size=10000
        )
        json.dump(scores, f, indent=2)

    logging.info(f"Done computing metrics {usage_and_time()}")

    save_script(output_path, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=["sentencepiece", "word", "char"],
        required=True,
    )
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_documents", type=int, default=0)
    parser.add_argument("-k", type=int, default=100)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=1.0)
    parser.add_argument(
        "--use_stop_words", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--ngram_range", nargs="+", type=int, default=(1, 1))
    parser.add_argument("--svd_components", type=int, default=100)

    args = parser.parse_args()

    if args.vectorizer == "sentencepiece" and args.tokenizer is None:
        raise ValueError("Tokenizer is required for sentencepiece vectorizer")

    if isinstance(args.ngram_range, list):
        args.ngram_range = tuple(args.ngram_range)

    main(args)
