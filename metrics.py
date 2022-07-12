import torch


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
