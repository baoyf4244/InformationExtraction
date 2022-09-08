# import json
#
# for key in ['train', 'dev']:
#     with open('data/kpe/{}.txt'.format(key), 'w', encoding='utf-8') as f_out:
#         with open('data/kpe/{}.json'.format(key), encoding='utf-8') as f_in:
#             prev = None
#             for line in f_in:
#                 line = json.loads(line)
#                 if line['label'] == "1":
#                     if prev is None or prev != line['abst']:
#                         processed_line = {'id': line['id'], 'text': line['abst'], 'keywords': line['keyword']}
#                         f_out.write(json.dumps(processed_line, ensure_ascii=False) + '\n')
#                         prev = line['abst']
# import json
#
# for key in ['train', 'dev', 'test']:
#     with open('data/nre/{}.txt'.format(key), 'w', encoding='utf-8') as f_out:
#         with open('data/nre/{}.sent'.format(key), encoding='utf-8') as f:
#             sents = f.readlines()
#
#         with open('data/nre/{}.pointer'.format(key), encoding='utf-8') as f:
#             pointer = f.readlines()
#
#         for sent, p in zip(sents, pointer):
#             data = {'text': sent.strip(), 'label': p.strip().split(' | ')}
#             f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
import json
from collections import defaultdict


def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output).strip().split()


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


# idx = 0
#
# for key in ['train', 'test']:
#     with open('data/{}.txt'.format(key), 'w', encoding='utf') as f_out:
#         with open('data/ner/{}.txt'.format(key), encoding='utf-8') as f:
#             for line in f:
#                 text = []
#                 labels = defaultdict(list)
#                 segments = line.strip().split()
#                 for segment in segments:
#                     phrase, label = segment.split('/')
#                     label = label.upper()
#                     tokens = tokenize_chinese_chars(phrase)
#                     if label == 'O':
#                         text += tokens
#                     else:
#                         labels[label].append([len(text), len(text) + len(tokens) - 1])
#                         text += tokens
#
#                 data = {
#                     'id': idx,
#                     'text': ' '.join(text),
#                     'labels': labels
#                 }
#                 f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
#
#                 idx += 1


# from collections import Counter
#
# counter = Counter()
# with open('data/train.txt', encoding='utf-8') as f:
#     for line in f:
#         line = json.loads(line)
#         text = line['text']
#         counter.update(text.split())
#
# with open('data/vocab.txt', 'w', encoding='utf-8') as f:
#     for k, v in counter.items():
#         if v >= 5:
#             f.write(k + '\n')


import torch
import torch.utils.data as tud
from tqdm.auto import tqdm
import warnings

def beam_search(
    model,
    X,
    predictions = 20,
    beam_width = 5,
    batch_size = 50,
    progress_bar = 0
):
    """
    Implements Beam Search to compute the output with the sequences given in X. The method can compute
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width.

    progress_bar: int
        Shows a tqdm progress bar, useful for tracking progress with large tensors. Ranges from 0 to 2.

    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the
        probability of the next token at every step.
    """
    with torch.no_grad():
        Y = torch.ones(X.shape[0], 1).to(next(model.parameters()).device).long()  # [bs, 1]
        # The next command can be a memory bottleneck, can be controlled with the batch
        # size of the predict method.
        next_probabilities = model.forward(X, Y)[:, -1, :]  # [bs, vs]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)  # [bs, bw]
        Y = Y.repeat((beam_width, 1))     # [bs * bw, 1]
        next_chars = next_chars.reshape(-1, 1)  # [bs * bw, 1]
        Y = torch.cat((Y, next_chars), axis = -1)   # [bs * bw, 2]
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            # X: [bs, ts] => [bw, bs, ts] => [bs, bw, ts] => [bs * bw, ts]
            dataset = tud.TensorDataset(X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1), Y)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)
            if progress_bar > 1:
                iterator = tqdm(iterator)
            for x, y in iterator:
                next_probabilities.append(model.forward(x, y)[:, -1, :].log_softmax(-1))
            next_probabilities = torch.cat(next_probabilities, dim=0)  # [bs * bw, vs]
            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            probabilities = probabilities.unsqueeze(-1) + next_probabilities  # [bs, bw, vs]
            probabilities = probabilities.flatten(start_dim = 1)  # [bs, bw * vs]
            probabilities, idx = probabilities.topk(k = beam_width, axis=-1)  # [bs, bw]
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(Y.shape[0] // beam_width, device = X.device).unsqueeze(-1) * beam_width
            Y = Y[best_candidates].flatten(end_dim = -2)
            Y = torch.cat((Y, next_chars), axis = 1)
        return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities


# from pytorch_beam_search import seq2seq
#
# # Create vocabularies
# # Tokenize the way you need
# source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
# target = [list("ABCDEFGHIJKL"), list("MNOPQRSTWXYZ")]
# # An Index object represents a mapping from the vocabulary
# # to integers (indices) to feed into the models
# source_index = seq2seq.Index(source)
# target_index = seq2seq.Index(target)
#
# # Create tensors
# X = source_index.text2tensor(source)
# Y = target_index.text2tensor(target)
# # X.shape == (n_source_examples, len_source_examples) == (2, 11)
# # Y.shape == (n_target_examples, len_target_examples) == (2, 12)
#
# # Create and train the model
# model = seq2seq.Transformer(source_index, target_index)    # just a PyTorch model
# model.fit(X, Y, epochs = 100)    # basic method included
#
# # Generate new predictions
# new_source = [list("new first in"), list("new second in")]
# new_target = [list("new first out"), list("new second out")]
# X_new = source_index.text2tensor(new_source)
# Y_new = target_index.text2tensor(new_target)
# loss, error_rate = model.evaluate(X_new, Y_new)    # basic method included
# predictions, log_probabilities = seq2seq.beam_search(model, X_new)
# output = [target_index.tensor2text(p) for p in predictions]
# output

import json
import matplotlib.pyplot as plt
from collections import Counter

counter = Counter()

with open('data/kpe/train.txt', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        counter[len(line['text'])] += 1

print(counter.most_common())
# counter = sorted(counter.items(), key=lambda k: k[0])
# plt.plot([c[0] for c in counter], [c[1] for c in counter])
# plt.show()
