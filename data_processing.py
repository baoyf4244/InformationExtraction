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


idx = 0

for key in ['train', 'test']:
    with open('data/{}.txt'.format(key), 'w', encoding='utf') as f_out:
        with open('data/ner/{}.txt'.format(key), encoding='utf-8') as f:
            for line in f:
                text = []
                labels = defaultdict(list)
                segments = line.strip().split()
                for segment in segments:
                    phrase, label = segment.split('/')
                    label = label.upper()
                    tokens = tokenize_chinese_chars(phrase)
                    if label == 'O':
                        text += tokens
                    else:
                        labels[label].append([len(text), len(text) + len(tokens) - 1])
                        text += tokens

                data = {
                    'id': idx,
                    'text': ' '.join(text),
                    'labels': labels
                }
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                idx += 1


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

