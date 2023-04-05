import logging
import os
import argparse
import sys
import json
from subprocess import check_output
from timeit import default_timer as timer
import resource

import numpy as np
import sentencepiece as spm

from tokenizers.implementations import (
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
)

logging.basicConfig(level=logging.INFO)


def save_script(output_path):
    """Save the script and arguments for reproducibility"""

    logging.info(f"Saving script and arguments at {output_path}")

    this_script = sys.argv[0]
    with open(this_script, "r") as cur_file:
        cur_running = cur_file.readlines()
    with open(os.path.join(output_path, this_script), "w") as log_file:
        log_file.writelines(cur_running)

    # get script name without extension
    script_name = os.path.splitext(os.path.basename(this_script))[0]

    with open(os.path.join(output_path, script_name + "_args.txt"), "w") as log_file:
        log_file.writelines([arg + "\n" for arg in sys.argv])


def get_output_path(
    output_dir,
    output_prefix,
    vocab_size,
    model_type,
    character_coverage,
    max_num_sentences,
    max_sentencepiece_length,
):
    if not max_num_sentences:
        additional = ""
    else:
        additional = f"_{max_num_sentences}sentences"

    if max_sentencepiece_length != 16:
        additional += f"_{max_sentencepiece_length}max_sentencepiece_length"

    output_path = os.path.join(
        output_dir,
        f"{output_prefix}_{model_type}_{vocab_size}vocab_{character_coverage}coverage{additional}",
    )

    return output_path


def train_huggingface_sentencepiece(
    files, output_path, vocab_size, model_type, max_num_sentences
):
    sp_special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    if model_type == "unigram":
        tokenizer = SentencePieceUnigramTokenizer()
    elif model_type == "bpe":
        tokenizer = SentencePieceBPETokenizer(unk_token="<unk>", fuse_unk=True)
    else:
        raise ValueError(f"Unknown tokenizer type: {model_type}.")

    class SubsampleLinesIterator:
        def __init__(self, files, max_num_sentences, seed=42):
            self.files = files
            self.max_num_sentences = max_num_sentences

            logging.info(f"Subsampling {max_num_sentences} sentences")
            start_time = timer()

            logging.info(
                f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
            )
            # Compute total number of sentences
            self.total_num_sentences = 0
            for file in self.files:
                num_lines = int(check_output(["wc", "-l", file]).split()[0])
                # with open(file, "r", encoding="utf-8") as f:
                #     for i, _ in enumerate(f):
                #         pass
                self.total_num_sentences += num_lines

            if self.max_num_sentences > self.total_num_sentences:
                self.max_num_sentences = self.total_num_sentences
                logging.warn(
                    f"max_num_sentences is larger than total number of sentences. Setting max_num_sentences to {self.max_num_sentences}"
                )

            # Pick random sentences
            self.random = np.random.RandomState(seed)
            self.random_sentences = self.random.choice(
                self.total_num_sentences, self.max_num_sentences, replace=False
            )
            self.random_sentences.sort()

            logging.info(f"Total number of sentences: {self.total_num_sentences}")
            logging.info(f"Number of sentences to use: {self.max_num_sentences}")

            # self.lines = []

            # logging.info(f"Number of sampled lines: {len(self.lines)}")
            logging.info(f"Subsampling took {timer() - start_time:.2f} seconds")
            logging.info(
                f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
            )

        def __iter__(self):
            line_num = 0
            rnd_sent_index = 0
            next_line_num = self.random_sentences[rnd_sent_index]

            for file in self.files:
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line_num == next_line_num:
                            # self.lines.append(line)
                            yield line
                            rnd_sent_index += 1
                            if rnd_sent_index < len(self.random_sentences):
                                next_line_num = self.random_sentences[rnd_sent_index]
                            else:
                                break
                        line_num += 1

    # Customize training
    logging.info(f"Training tokenizer on:\n{files}")
    start_time = timer()
    additional_args = {
        "vocab_size": vocab_size,
        "show_progress": True,
        "special_tokens": sp_special_tokens,
    }
    if model_type == "unigram":
        additional_args["unk_token"] = "<unk>"

    if max_num_sentences:
        tokenizer.train_from_iterator(
            SubsampleLinesIterator(files, max_num_sentences), **additional_args
        )
    else:
        logging.info(f"Using all sentences")
        tokenizer.train(files, **additional_args)

    logging.info(f"Training took {timer() - start_time:.2f} seconds")

    logging.info(f"Saving tokenizer at {output_path}")

    tokenizer.save(os.path.join(output_path, "tokenizer.json"))

    logging.info(
        f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )
    # Saving vocab
    with open(
        os.path.join(output_path, "vocab.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(
            dict(sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])),
            outfile,
            indent=2,
            ensure_ascii=False,
        )

    logging.info("Done creating tokenizer")


def train_original_sentencepiece(
    input,
    num_threads,
    model_prefix,
    vocab_size,
    character_coverage,
    model_type,
    input_sentence_size,
    max_sentencepiece_length,
):
    start_time = timer()
    logging.info(
        f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )
    spm.SentencePieceTrainer.train(
        input=input,
        num_threads=num_threads,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        train_extremely_large_corpus=True,
        shuffle_input_sentence=True,
        input_sentence_size=input_sentence_size,
        max_sentencepiece_length=max_sentencepiece_length,
    )
    logging.info(f"Training took {timer() - start_time:.2f} seconds")
    logging.info(
        f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )


def main(args):
    output_path = get_output_path(
        args.output_dir,
        args.output_prefix,
        args.vocab_size,
        args.model_type,
        args.character_coverage,
        args.max_num_sentences,
        args.max_sentencepiece_length,
    )

    sentencepiece_trained = not args.huggingface and os.path.exists(
        os.path.join(output_path, "m.model")
    )
    huggingface_trained = args.huggingface and os.path.exists(
        os.path.join(output_path, "tokenizer.json")
    )
    if sentencepiece_trained or huggingface_trained:
        logging.error(f"Tokenizer already exists at {output_path}")
        if args.overwrite:
            logging.info("Overwriting...")
        else:
            logging.info("Exiting...")
            sys.exit(1)

    os.makedirs(output_path, exist_ok=True)

    if args.huggingface:
        if args.character_coverage:
            raise ValueError(
                "character_coverage is not supported for huggingface tokenizer"
            )
        if args.max_sentencepiece_length:
            raise ValueError(
                "max_sentencepiece_length is not supported for huggingface tokenizer"
            )

        logging.info("Training huggingface sentencepiece")
        train_huggingface_sentencepiece(
            args.input,
            output_path,
            args.vocab_size,
            args.model_type,
            args.max_num_sentences,
        )

    else:
        logging.info("Training original sentencepiece")
        train_original_sentencepiece(
            input=args.input,
            num_threads=args.num_threads,
            model_prefix=os.path.join(output_path, "m"),
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type,
            input_sentence_size=args.max_num_sentences,
            max_sentencepiece_length=args.max_sentencepiece_length,
        )

    save_script(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--model_type", type=str, choices=["bpe", "unigram"])
    parser.add_argument("--character_coverage", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_num_sentences", type=int, default=0)
    parser.add_argument(
        "--huggingface", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--max_sentencepiece_length", type=int, default=16)

    args = parser.parse_args()
    main(args)
