import torch
import numpy as np

def num_correct(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), 1, True, True)
        correct = pred.eq(target.unsqueeze(1).expand_as(pred))

        num_correct = np.empty_like(topk, dtype=np.int64)
        for idx, k in enumerate(topk):
            correct_k = correct[:, :k].sum()
            num_correct[idx] = correct_k.item()
        return num_correct
