import torch
from collections import defaultdict

def mrc_span_f1(start_logits, end_logits, span_logits, start_masks, end_masks, span_labels):
    """
    Compute span f1 according to query-based model output
    Args:
        start_logits: [bsz, seq_len]
        end_logits: [bsz, seq_len]
        span_logits: [bsz, seq_len, seq_len]
        start_masks: [bsz, seq_len]
        end_masks: [bsz, seq_len]
        span_labels: [bsz, seq_len, seq_len]
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """

    start_preds = start_logits > 0
    end_preds = end_logits > 0
    span_preds = span_logits > 0

    bsz, seq_len = start_masks.size()

    start_masks = start_masks.bool()
    end_masks = end_masks.bool()
    span_labels = span_labels.bool()

    start_preds = start_preds.bool()
    end_preds = end_preds.bool()

    span_preds = (span_preds
                  & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                  & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    span_label_mask = (start_masks.unsqueeze(-1).expand(-1, -1, seq_len)
                       & end_masks.unsqueeze(1).expand(-1, seq_len, -1))
    span_label_mask = torch.triu(span_label_mask, 0)  # start should be less or equal to end
    span_preds = span_label_mask & span_preds

    tp = (span_labels & span_preds).long().sum()
    fp = (~span_labels & span_preds).long().sum()
    fn = (span_labels & ~span_preds).long().sum()
    return tp, fp, fn


def get_f1_score(tp, fp, fn):
    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * recall * precision / (recall + precision + 1e-10)
    return recall, precision, f1


def get_ner_labels(tag_ids, masks, idx2tag):
    """
    根据标签ID返回标签名称对应的索引位置，标签采用BIO模式
    :param masks:
    :param tag_ids: 标签ID
    :param idx2tag: 标签ID对应的标签名称
    :return: dict，标签名称及相应的索引列表，索引列表为二元组，元组第一个元素为实体起始位置，第二个索引为实体结束位置+1（方便取值）
    """
    if isinstance(tag_ids, torch.Tensor):
        tag_ids = tag_ids.numpy().tolist()

    if isinstance(masks, torch.Tensor):
        seq_len = masks.sum()
    else:
        seq_len = sum(masks)
    tags = [idx2tag[tag_id] for tag_id in tag_ids]
    labels = defaultdict(set)
    i = 0
    while i < seq_len:
        if tags[i].startswith('B-'):
            label = tags[i].split('-')[1]
            start = i
            i += 1
            while i < seq_len and tags[i].startswith('I-') and tags[i].split('-')[1] == label:
                i += 1
            labels[label].add((start, i))
        else:
            i += 1

    return labels


def flat_ner_stats(preds, targets, masks, idx2tag):
    tp, fp, fn = 0, 0, 0
    for pred, target, mask in zip(preds, targets, masks):
        pred_labels = get_ner_labels(pred, mask, idx2tag)
        target_labels = get_ner_labels(target, mask, idx2tag)
        for label, indices in target_labels.items():
            pred_indices = pred_labels[label]
            tp += len(pred_indices.intersection(indices))

            for idx in pred_indices:
                if idx not in indices:
                    fp += 1

            for idx in indices:
                if idx not in pred_indices:
                    fn += 1
    return tp, fp, fn


if __name__ == '__main__':
    id2tag_test = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
    tag_ids_test = torch.LongTensor([[0, 1, 2, 2, 1, 2, 1, 4, 0, 3, 4, 4, 0, 0, 3]])
    tag_ids_target = torch.LongTensor([[0, 1, 2, 2, 1, 2, 2, 0, 0, 3, 4, 4, 0, 0, 3]])
    masks_test = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    # print(get_ner_labels(tag_ids_test, id2tag_test))
    print(flat_ner_stats(tag_ids_test, tag_ids_target, masks_test, id2tag_test ))