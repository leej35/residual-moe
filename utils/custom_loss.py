import torch

def mae_rmse(pred, trg, mask_neg=False, denom=None):
    # inputs: torch 1d array
    if mask_neg:
        neg_mask = (trg >= 0)
        if pred.is_cuda:
            device = pred.get_device()
            neg_mask = neg_mask.to(device)

        # NOTE: behavior of torch.masked_select is to extract only elements
        # with value=1 in its mask tensor. If val=0, it will be disregarded.
        pred = torch.masked_select(pred, neg_mask)
        trg = torch.masked_select(trg, neg_mask)
    if denom is not None:
        mae = torch.sum(abs(pred - trg)) / denom
        rmse = torch.sqrt(torch.sum(torch.pow((trg - pred), 2)) / denom)

    else:
        mae = torch.mean(abs(pred - trg))
        rmse = torch.sqrt(torch.mean(torch.pow((trg - pred), 2)))

    return mae, rmse


class Logit(torch.nn.Module):
    def forward(self, input):
        return torch.log(input) - torch.log(1 - input)


