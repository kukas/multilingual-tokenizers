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

logging.basicConfig(level=logging.INFO)

from notebook_utils import SpTokenizer
from notebook_utils import compute_counters
from collections import Counter
from notebook_utils import counter_to_freqs


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time
from sklearn.cluster import MiniBatchKMeans

from scipy.sparse import csr_matrix


def encode(tokenizer, doc):
    result = []
    for lines in doc:
        result += tokenizer.encode(lines)
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
        defaults={"k": 100, "max_documents": None, "token_freq_cutoff": 2},
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
    original_data = [doc for docs in original_data for doc in docs]

    merged_tokenizer = SpTokenizer(args.tokenizer)

    logging.info(f"Tokenizing data {usage_and_time()}")
    # tokenized_docs = one document per element. Each element is a list of tokens
    tokenized_docs = parallel_list_map(partial(encode, merged_tokenizer), original_data)

    logging.info(f"Count token occurrences {usage_and_time()}")
    counter_per_document = parallel_list_map(Counter, tokenized_docs)

    logging.info(f"Count overall token occurrences {usage_and_time()}")
    counter_overall = Counter()
    for counter in counter_per_document:
        counter_overall.update(counter)

    freqs_overall = counter_to_freqs(
        counter_overall, vocab_size=len(merged_tokenizer.vocab)
    )

    vocab_mapping = {}
    j = 0
    for i in np.argwhere(freqs_overall > args.token_freq_cutoff)[:, 0]:
        # skip unknown tokens
        if i == 0:
            continue
        vocab_mapping[i] = j
        j += 1

    logging.info(f"Number of features for each document: {len(vocab_mapping)}")
    logging.info(f"Number of documents: {len(original_data)}")

    logging.info(f"Computing features {usage_and_time()}")

    term_frequency = csr_matrix(
        (len(original_data), len(vocab_mapping)), dtype=np.float64
    )
    # term_occurence = csr_matrix((len(original_data), len(vocab_mapping)), dtype=np.int8)

    i = 0
    term_frequency_data = []
    term_frequency_row_ind = []
    term_frequency_col_ind = []
    for counter in counter_per_document:
        for token, count in counter.items():
            row = []
            if token in vocab_mapping:
                # term_frequency[i, vocab_mapping[token]] = count
                row.append(count)
                term_frequency_row_ind.append(i)
                term_frequency_col_ind.append(vocab_mapping[token])
            row_sum = sum(row)
            if row_sum > 0:
                row = [x / row_sum for x in row]
            term_frequency_data += row
            # term_occurence[i, vocab_mapping[token]] = 1
        # row_sum = term_frequency[i].sum()
        # if row_sum > 0:
        #     term_frequency[i] /= term_frequency[i].sum()
        i += 1

    # document_count = np.sum(term_occurence, axis=0, dtype=np.int32)
    # inverse_document_frequency = np.log(len(original_data) / document_count)

    # tf_idf = term_frequency * inverse_document_frequency

    logging.info(f"Convert to csr matrix {usage_and_time()}")
    features = csr_matrix(
        (term_frequency_data, (term_frequency_row_ind, term_frequency_col_ind)),
        shape=(len(original_data), len(vocab_mapping)),
    )

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

    save_script(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_documents", type=int, default=0)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--token_freq_cutoff", type=int, default=2)

    args = parser.parse_args()
    main(args)
