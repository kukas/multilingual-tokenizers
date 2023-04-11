import logging
import os
import argparse
import sys
import json
import re
import random
from collections import defaultdict
from subprocess import check_output
from timeit import default_timer as timer
import resource

# allow import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sentencepiece as spm

from functools import partial

from utils import save_script, get_output_path
from notebook_utils import parallel_list_map, load_lines

from notebook_utils import SpTokenizer
from notebook_utils import compute_counters
from notebook_utils import counter_to_freqs

from sentencepiece import sentencepiece_model_pb2 as model

from time import time

import multiprocessing

logging.basicConfig(level=logging.INFO)


def encode(tokenizer, lines):
    return tokenizer.encode(lines, out_type=str)


from notebook_utils import merge_tokenizer_protos, save_tokenizer, trim_vocab


from timeit import default_timer as timer

last_time = timer()
grouped_tokenizers = None


def usage_and_time():
    global last_time
    message = f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB, Time: {timer() - last_time:.2f} s"
    last_time = timer()
    return message


def dummy(x):
    global grouped_tokenizers
    return x, x, x, len(grouped_tokenizers)


def _construct_tokenizer(grouped_tokenizers, pointers):
    tokenizers = []
    for group, toks in grouped_tokenizers.items():
        tokenizers.append(toks[pointers[group]])
    return merge_tokenizer_protos(tokenizers)


def try_merge(group, pointers, compute_metric):
    global grouped_tokenizers
    new_pointers = pointers.copy()
    new_pointers[group] += 1
    if new_pointers[group] >= len(grouped_tokenizers[group]):
        logging.info(f"Skipping {group} because it is at the end")
        return np.inf, None, None, None

    new_tokenizer_proto = _construct_tokenizer(grouped_tokenizers, new_pointers)
    new_tokenizer = spm.SentencePieceProcessor(
        model_proto=new_tokenizer_proto.SerializeToString()
    )
    new_metric = compute_metric(new_tokenizer)
    return (new_metric, new_pointers, len(new_tokenizer_proto.pieces), group)


# def _compute_entropy(tokenizer, data):
#     entropies = [
#         tokenizer.CalculateEntropy(lines, alpha=0) for lines in original_data
#     ]
#     return np.mean(list(map(np.mean, entropies)))


def _compute_mean_sentence_length(tokenizer, data):
    lengths = [[len(tokens) for tokens in tokenizer.encode(lines)] for lines in data]
    return np.mean(list(map(np.mean, lengths)))


def _sample_data(data, sample_lines):
    if args.sample_lines > 0:
        return [
            random.sample(lines, args.sample_lines)
            if len(lines) > sample_lines
            else lines
            for lines in data
        ]
    else:
        return data


def main(args):
    global grouped_tokenizers
    output_path = get_output_path(
        args.output_dir,
        args.output_prefix,
        args={
            "max_lines": args.max_lines,
            "sample_lines": args.sample_lines,
            "target_vocab_size": args.target_vocab_size,
        },
        defaults={"max_lines": 0, "sample_lines": 0, "target_vocab_size": 120000},
        always_include=["target_vocab_size"],
    )
    os.makedirs(output_path, exist_ok=True)
    print(output_path)

    logging.info(
        f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )
    logging.info(f"Loading data")
    original_data = [load_lines(file, max_lines=args.max_lines) for file in args.input]
    logging.info(f"Done loading data {usage_and_time()}")

    # original_data = [doc for docs in original_data for doc in docs]
    assert args.group_regex is not None or args.tokenizer_groups is not None

    if args.group_regex is None:
        tokenizer_groups = args.tokenizer_groups
    else:
        tokenizer_groups = []
        for tok_path in args.tokenizers:
            group = re.search(args.group_regex, tok_path)
            # print(tok_path, group)
            tokenizer_groups.append(group.group(0).strip())

    assert len(args.tokenizers) == len(
        tokenizer_groups
    ), "Must have one tokenizer group per tokenizer"

    logging.info(f"Loading tokenizers")

    tokenizers = []
    # collect all tokens and their scores
    for path in args.tokenizers:
        m = model.ModelProto()
        model_path = os.path.join(path, "m.model")
        if os.path.exists(model_path):
            m.ParseFromString(open(model_path, "rb").read())
            tokenizers.append(m)
        else:
            logging.info(f"Skipping {path} because it does not exist")
            tokenizers.append(None)
        # m.ParseFromString(open(model_path, "rb").read())

    grouped_tokenizers = defaultdict(list)
    for group, tok in zip(tokenizer_groups, tokenizers):
        if tok is not None:
            grouped_tokenizers[group].append(tok)

    for group, toks in grouped_tokenizers.items():
        # print([len(tok.pieces) for tok in toks])
        grouped_tokenizers[group] = list(sorted(toks, key=lambda x: len(x.pieces)))
        # print([len(tok.pieces) for tok in grouped_tokenizers[group]])

    pointers = {group: 0 for group in grouped_tokenizers.keys()}

    logging.info(f"Done loading tokenizers {usage_and_time()}")

    logging.info(f"Construct first merged tokenizer")

    merged_tokenizer_proto = _construct_tokenizer(grouped_tokenizers, pointers)
    # merged_tokenizer = SpTokenizer(model_proto=merged_tokenizer.SerializeToString())
    merged_tokenizer = spm.SentencePieceProcessor(
        model_proto=merged_tokenizer_proto.SerializeToString()
    )

    logging.info(f"Done constructing first merged tokenizer {usage_and_time()}")

    logging.info(f"Computing initial metric value")

    # tokenized_docs = parallel_list_map(partial(encode, merged_tokenizer), original_data)

    sampled_data = _sample_data(original_data, args.sample_lines)
    metric = _compute_mean_sentence_length(merged_tokenizer, sampled_data)
    print(metric)

    logging.info(f"Done computing metric {usage_and_time()}")

    # print(grouped_tokenizers.keys())

    target_vocab_size = args.target_vocab_size
    merged_tokenizer_vocab_size = len(merged_tokenizer_proto.pieces)
    iters = 0

    while merged_tokenizer_vocab_size < target_vocab_size:
        # candidates = []
        # logging.info(f"Adding capacity iteration {iters}")
        # for i, group in enumerate(grouped_tokenizers.keys()):
        #     new_pointers = pointers.copy()
        #     new_pointers[group] += 1
        #     if new_pointers[group] >= len(grouped_tokenizers[group]):
        #         continue
        #     new_tokenizer_proto = _construct_tokenizer(grouped_tokenizers, new_pointers)
        #     new_tokenizer = spm.SentencePieceProcessor(
        #         model_proto=new_tokenizer_proto.SerializeToString()
        #     )
        #     new_metric = _compute_mean_sentence_length(new_tokenizer, original_data)
        #     candidates.append((new_metric, new_pointers, new_tokenizer_proto, group))
        #     print("add capacity for", group, new_metric)
        #     print("new vocab size", len(new_tokenizer_proto.pieces))
        # candidates = sorted(candidates, key=lambda x: x[0])
        # metric, pointers, merged_tokenizer_proto, best_group = candidates[0]
        # print("best", best_group, metric)
        # logging.info(f"Done adding capacity iteration {iters} {usage_and_time()}")
        # iters += 1

        logging.info(f"Adding capacity iteration {iters}")
        sampled_data = _sample_data(original_data, args.sample_lines)
        compute_metric = partial(_compute_mean_sentence_length, data=sampled_data)
        # compute_metric = dummy
        generate_candidate = partial(
            try_merge,
            pointers=pointers,
            compute_metric=compute_metric,
        )
        # generate_candidate = dummy
        candidates = parallel_list_map(
            generate_candidate,
            grouped_tokenizers.keys(),
            n_jobs=multiprocessing.cpu_count(),
        )
        # print(candidates)
        min_candidate = min(candidates, key=lambda x: x[0])
        metric, pointers, merged_tokenizer_vocab_size, best_group = min_candidate
        print("best", best_group, metric, merged_tokenizer_vocab_size)
        logging.info(f"Done adding capacity iteration {iters} {usage_and_time()}")
        iters += 1

    logging.info(f"Done merging tokenizers {usage_and_time()}")
    logging.info(f"Generate merged tokenizer")
    pointers[best_group] += 1
    merged_tokenizer_proto = _construct_tokenizer(grouped_tokenizers, pointers)

    logging.info(f"Trim vocab")
    merged_tokenizer_proto = trim_vocab(merged_tokenizer_proto, target_vocab_size)
    assert len(merged_tokenizer_proto.pieces) == target_vocab_size

    save_tokenizer(merged_tokenizer_proto, output_path)
    save_script(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--tokenizer_groups", nargs="+")
    parser.add_argument("--group_regex", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--sample_lines", type=int, default=0)
    parser.add_argument("--target_vocab_size", type=int, default=120000)

    args = parser.parse_args()
    main(args)
