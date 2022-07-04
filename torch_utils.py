import torch


def sequence_mask(tensor, max_len):
    if len(tensor.size()) == 1:
        tensor = torch.unsqueeze(tensor, 1)
    batch_size = tensor.size(0)
    masked_helper = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = masked_helper < tensor
    return mask


if __name__ == '__main__':
    a = torch.randint(1, 7, [10])
    print(a)
    print(sequence_mask(a, 7))
