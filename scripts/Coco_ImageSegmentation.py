import torch.nn as nn
import torch
import transformers

class pixel_loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def false_negative_rate(self, preds, ys):
        # preds: [C, Num_classes, H, W]
        # ys: [C, H, W]
        categories = torch.zeros_like(ys)
        categories = preds.argmax(dim=1)

        correct = (categories == ys)
        neg_rates = torch.zeros(self.num_classes)
        for ix in self.num_classes:
            n_total = (ys == ix).sum()
            neg_rates[ix] = (ys==ix & correct).sum()/n_total

        return torch.mean(neg_rates)

