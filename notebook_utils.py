import math
import sentencepiece as spm


class SpTokenizer:
    def __init__(self, model_file=None, model_proto=None):
        if model_file is None and model_proto is None:
            raise ValueError("Either model_file or model_proto must be provided")

        self.sp_model = spm.SentencePieceProcessor(
            model_file=model_file, model_proto=model_proto
        )
        self.vocab = [self.sp_model.IdToPiece(idx) for idx in range(len(self.sp_model))]

    def encode(self, text, **kwargs):
        return self.sp_model.encode(text, **kwargs)


def get_vocab_from_sp_model(sp):
    return [sp.id_to_piece(i) for i in range(len(sp))]


# format function for the numbers
def humanize(n, base=1000):
    decimalnames = ["", "K", "M", "B"]
    binarynames = ["", "KiB", "MiB", "GiB"]
    names = binarynames if base == 1024 else decimalnames
    n = float(n)
    idx = max(
        0,
        min(
            len(names) - 1,
            int(math.floor(0 if n == 0 else math.log(abs(n)) / math.log(base))),
        ),
    )

    return "{:.1f}{}".format(n / base ** (idx), names[idx])


def flatten(l):
    return [item for sublist in l for item in sublist]


def read_vocab(vocab_file, return_probs=False):
    with open(vocab_file) as tsv_file:
        first_column = []
        second_column = []

        for line in tsv_file:
            fst, snd = line.rsplit("\t", 1)
            first_column.append(fst)
            second_column.append(float(snd))
    if return_probs:
        return first_column, second_column
    return first_column


def load_lines(path, max_lines=None, filter_empty=True):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines is not None and max_lines != 0 and len(lines) >= max_lines:
                break
            line = line.rstrip()
            if filter_empty and not line:
                continue
            lines.append(line)

    return lines


def load_documents(path, max_documents=None):
    docs = []
    with open(path) as f:
        lines = []
        for i, line in enumerate(f):
            if (
                max_documents is not None
                and max_documents != 0
                and len(docs) >= max_documents
            ):
                break

            if line == "\n":
                lines = " ".join(lines)
                docs.append(lines)
                lines = []
            else:
                lines.append(line.rstrip())

    return docs


def load_document_lines(path, max_documents=None):
    docs = []
    with open(path) as f:
        lines = []
        for i, line in enumerate(f):
            if (
                max_documents is not None
                and max_documents != 0
                and len(docs) >= max_documents
            ):
                break

            if line == "\n":
                docs.append(lines)
                lines = []
            else:
                lines.append(line.rstrip())

    return docs


import lzma


def open_lzma(path, max_docs):
    docs = []
    with lzma.open(path, mode="rt") as f:
        doc = []
        for line in f:
            if line == "\n":
                docs.append(doc)
                doc = []
            else:
                doc.append(line.strip())
            if len(docs) >= max_docs:
                break
    return docs


# def load_documents(path, max_lines=None, filter_empty=True):
#     lines = []
#     with open(path) as f:
#         for i, line in enumerate(f):
#             if max_lines is not None and len(lines) >= max_lines:
#                 break
#             line = line.rstrip()
#             if filter_empty and len(line.strip()) == 0:
#                 continue
#             lines.append(line)

#     return lines


import numpy as np


def compute_sorted_token_freqs(tokenized_text, vocabulary):
    # returns a list of tuples (freq, token)
    # and the frequency of the <unk> token

    freqs = [0] * len(vocabulary)
    for sentence in tokenized_text:
        for token in sentence:
            freqs[token] += 1
    assert vocabulary[0] == "<unk>"
    sorted_freqs = sorted(list(zip(freqs[1:], vocabulary[1:])), reverse=True)
    return sorted_freqs, freqs[0]


def compute_freqs(tokenized_text, vocab_size, return_probs=False):
    # freqs = np.zeros(vocab_size, dtype=np.int32)
    freqs = [0] * vocab_size
    for sentence in tokenized_text:
        for token in sentence:
            freqs[token] += 1

    if return_probs:
        freqs = freqs / freqs.sum()

    return freqs


from collections import Counter


def compute_counters(tokenized_text):
    counters = []
    for line in tokenized_text:
        counter = Counter(line)
        counters.append(dict(counter))

    return counters


# def compute_counters_docs(tokenized_docs):
#     counters = []
#     for lines in tokenized_docs:
#         counter = Counter()
#         for line in lines:
#             counter.update(line)
#         counters.append(dict(counter))

#     return counters


def counter_to_freqs(counter, vocab_size):
    freqs = np.zeros(vocab_size, dtype=np.int32)
    for token, freq in counter.items():
        freqs[token] += freq
    return freqs


def compute_probs(freqs):
    return freqs / freqs.sum()


def compute_characters_per_token(tokenized_text, original_text, vocab):
    # caveat: adds up also unknown tokens
    total_tokens = 0
    total_chars = 0
    lengths = [0] * 32
    for token_sent, orig_sent in zip(tokenized_text, original_text):
        total_tokens += len(token_sent)
        total_chars += len(orig_sent)
        for token in token_sent:
            decoded = vocab[token]
            lengths[len(decoded)] += 1
        # for token in sentence:
        #     total_chars += len(token)
    return total_chars / total_tokens, lengths


def compute_characters_per_token_from_freqs(freqs, vocab, count_unk=False):
    total_tokens = 0
    total_chars = 0
    for token, freq in zip(vocab, freqs):
        if token == "<unk>":
            if count_unk:
                total_tokens += freq
                # total_chars += freq * 0  # count unk as 0 character (punishes cpt)
        else:
            total_tokens += freq
            total_chars += freq * len(token)
    return total_chars / total_tokens


def compute_average_rank(probs):
    probs = np.sort(probs)[::-1]
    ranks = np.arange(1, len(probs) + 1)
    return np.average(ranks, weights=probs)


def word_tokenize_list(lines, language="english", preserve_line=False):
    import nltk
    nltk.download("punkt")
    from nltk.tokenize import word_tokenize

    return [
        word_tokenize(line, language=language, preserve_line=False) for line in lines
    ]


def tokenize_words(words_text, tokenizer):
    tokens_per_word = []
    for sentence in words_text:
        tokens_per_word.append([])
        for i, word in enumerate(sentence):
            tokens_per_word[-1].append(tokenizer.encode(word))
    return tokens_per_word


# def compute_fertility(tokenized_text, original_text):
#     # subword fertility measures the average number of subwords produced per tokenized word.
#     # A minimum fertility of 1 means that the tokenizer’s vocabulary contains every single word in the text.
#     total_tokens = 0
#     total_words = 0
#     for token_sent, orig_sent in zip(tokenized_text, original_text):
#         words = word_tokenize(orig_sent, language, preserve_line=True)
#         total_words += len(words)
#         total_tokens += len(token_sent)
#     return total_tokens/total_words


def compute_fertility(words_text, tokenized_words_text):
    total_words = 0
    total_tokens = 0
    for words_sent, tokenized_words_sent in zip(words_text, tokenized_words_text):
        assert len(words_sent) == len(tokenized_words_sent)
        total_words += len(words_sent)
        total_tokens += sum([len(tokens) for tokens in tokenized_words_sent])

    return total_tokens / total_words


assert compute_fertility([["hello", "world"]], [[[1], [2]]]) == 1.0
assert compute_fertility([["hello", "world"]], [[[1, 2], [3, 4]]]) == 2.0


def compute_proportion_of_continued_words(words_text, tokenized_words_text):
    total_words = 0
    total_continued_words = 0
    for words_sent, tokenized_words_sent in zip(words_text, tokenized_words_text):
        assert len(words_sent) == len(tokenized_words_sent)
        total_words += len(words_sent)
        for word, tokens in zip(words_sent, tokenized_words_sent):
            if len(tokens) > 1:
                total_continued_words += 1
    return total_continued_words / total_words


assert compute_proportion_of_continued_words([["hello", "world"]], [[[1], [2]]]) == 0.0
assert (
    compute_proportion_of_continued_words([["hello", "world"]], [[[1, 2], [3, 4]]])
    == 1.0
)
assert (
    compute_proportion_of_continued_words([["hello", "world"]], [[[1, 2], [3]]]) == 0.5
)


def compute_average_log_probability(freqs, num_sentences):
    probs = compute_probs(freqs)
    # return np.sum(freqs * np.log(probs)) / np.sum(freqs)
    return np.sum(freqs * np.log(probs)) / num_sentences


def compute_entropy(probs):
    return -np.sum(probs * np.log(probs))


def compute_F_at_95(sorted_freqs):
    probs = compute_probs(sorted_freqs)
    cumsum = np.cumsum(probs)
    idx = np.where(cumsum > 0.95)[0][0]
    return sorted_freqs[idx]


def tokens_to_cover_95(probs):
    assert np.isclose(np.sum(probs), 1)
    probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(probs)
    idx = np.where(cumsum > 0.95)[0][0]
    return idx


def compute_divergence_from_uniform(freqs):
    p_uniform = 1 / len(freqs)
    probs = compute_probs(freqs)
    return np.sum(np.abs(probs - p_uniform)) / 2


from scipy.spatial.distance import jensenshannon


def compute_jsd(freqs1, freqs2):
    probs1 = compute_probs(freqs1)
    probs2 = compute_probs(freqs2)
    return jensenshannon(probs1, probs2)


# def compute_alp_zheng(lines, tokenizer):
#     all_tokens = 0
#     words_list = tokenizer.vocab
#     words = {}
#     for i, word in enumerate(words_list):
#         words[i] = 0

#     tokenized_lines = []
#     for line in lines:
#         line = line.strip()
#         token_ids = tokenizer.tokenize(line)
#         all_tokens += len(token_ids)
#         for idx in token_ids:
#             words[idx] += 1
#         tokenized_lines.append(token_ids)
#     for idx in words.keys():
#         words[idx] /= all_tokens
#     probs = []
#     for token_ids in tokenized_lines:
#         p = 0.0
#         for idx in token_ids:
#             p += math.log(words[idx])
#         probs.append(p)

#     return np.mean(probs)
# Note: I checked it to be equivalent to compute_average_log_probability


import os
from glob import glob
from sentencepiece import sentencepiece_model_pb2 as model
from collections import defaultdict
from copy import deepcopy


def merge_tokenizer_protos(model_protos, log_scores=True):
    tokens_scores = defaultdict(list)
    num_tokenizers = len(model_protos)

    # collect all tokens and their scores
    for m in model_protos:
        scores = []
        for piece in m.pieces:
            # skip special tokens
            if piece.type != 1:
                continue
            tokens_scores[piece.piece].append(piece.score)

    # normalize the scores
    for token, scores in tokens_scores.items():
        if log_scores:
            if len(scores) != num_tokenizers:
                scores = scores + [-np.inf] * (num_tokenizers - len(scores))
            tokens_scores[token] = np.exp(scores).sum() / num_tokenizers
            tokens_scores[token] = np.log(tokens_scores[token])
        else:
            scores = scores + [-len(tokens_scores)] * (num_tokenizers - len(scores))
            tokens_scores[token] = sum(scores) / num_tokenizers

    tokens_scores = list(
        sorted(tokens_scores.items(), key=lambda item: item[1], reverse=True)
    )

    # create a new tokenizer based on one of the old tokenizers
    m = deepcopy(model_protos[0])
    old_pieces = deepcopy(m.pieces)
    m.ClearField("pieces")

    for piece in old_pieces:
        if piece.type != 1:
            m.pieces.append(piece)

    for token, score in tokens_scores:
        piece = model.ModelProto().SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = 1
        m.pieces.append(piece)
    m.trainer_spec.vocab_size = len(m.pieces)

    return m


def merge_tokenizers(paths, log_scores=True):
    protos = []
    # collect all tokens and their scores
    for path in paths:
        m = model.ModelProto()
        m.ParseFromString(open(path, "rb").read())
        protos.append(m)

    return merge_tokenizer_protos(protos, log_scores)


def save_tokenizer(m, output_dir, name="m"):
    os.makedirs(output_dir, exist_ok=True)
    print("Saving tokenizer to", output_dir)
    with open(os.path.join(output_dir, f"{name}.model"), "wb") as f:
        f.write(m.SerializeToString())
    with open(os.path.join(output_dir, f"{name}.vocab"), "w") as f:
        for piece in m.pieces:
            f.write(piece.piece + "\t" + str(piece.score) + "\n")


def trim_vocab(m, vocab_size):
    assert vocab_size <= len(m.pieces)
    orig_pieces = deepcopy(m.pieces)
    m.ClearField("pieces")
    for i in range(vocab_size):
        m.pieces.append(orig_pieces[i])
    m.trainer_spec.vocab_size = len(m.pieces)
    return m


from functools import partial
from multiprocessing import Pool


def _map(func, key, value):
    return (key, func(value))


def parallel_dict_map(func, d, n_jobs=16):
    with Pool(n_jobs) as p:
        return dict(p.starmap(partial(_map, func), d.items()))


def parallel_list_map(func, d, n_jobs=16):
    with Pool(n_jobs) as p:
        return list(p.map(func, d, chunksize=math.ceil(len(d) / n_jobs)))


def create_tokenizer(tuple_pieces):
    # tokens_scores = defaultdict(list)
    # num_tokenizers = len(model_protos)

    # create a new tokenizer based on one of the old tokenizers
    # m = deepcopy(model_protos[0])
    m = model.ModelProto()
    # old_pieces = deepcopy(m.pieces)
    # m.ClearField("pieces")

    # for piece in old_pieces:
    #     if piece.type != 1:
    #         m.pieces.append(piece)

    for token, score in tuple_pieces:
        piece = model.ModelProto().SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = 1
        if token == "<unk>":
            piece.type = 2
        if token == "<s>" or token == "</s>":
            piece.type = 3
        m.pieces.append(piece)
    m.trainer_spec.vocab_size = len(m.pieces)

    return m
