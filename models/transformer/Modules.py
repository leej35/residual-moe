import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Code based on https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

class ScaledDotProductAttention4d(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, transpose_dim=(4, 3)):
        """
        attn_mode:
            - 'merged' : merge n_timestep and n_events dimension
            - 'separate' : non-merge

        q : (n_b x n_heads x n_timestep x n_event x d_k)
        k : (n_b x n_heads x n_timestep x n_event x d_k)
        -> k.transpose(4,3) : (n_b x n_heads x n_timestep x d_k x n_event)

        **current
        matmul(q,k) -> (n_b x n_heads x n_timestep x n_event x n_event)

        ** variants : merge n_times and n_event
        q_merged = q.view(q.size(0), q.size(1), -1, q.size(-1))     
        k_merged = k.view(k.size(0), k.size(1), -1, k.size(-1))        
        
        torch.matmul(q_merged, k_merged.transpose(2,3)) -> (n_b x n_heads x merged_dim x merged_dim); 
            - merged_dim = n_events * n_timesteps
         
        NOTE: for non-merge mode, use transpose_dim = (4, 3)
        """

        attn = torch.matmul(q / self.temperature, k.transpose(transpose_dim))

        if mask is not None:
            if attn.dtype == torch.float16:
                attn = attn.masked_fill(mask == 0, -65000).type_as(attn)
            else:
                attn = attn.masked_fill(mask == 0, -1e30).type_as(attn)

            # mask: (n_batch x 1 (=n_heads;for broadcasting) x n_steps x n_event x n_event) 
            # NOTE: last dim

        # NOTE: dim for Softmax is where we can also have different projections 
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q,k,v : (n_batch x (n_seqlen x n_event) x n_head x d_k)
        mask:   (n_batch x 1 x seq_len x n_event x n_event)
        """

        # attn : (n_batch x (n_seqlen x n_event) x (n_seqlen x n_event))
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            if attn.dtype == torch.float16:
                attn.masked_fill_(mask == 0, -65000).type_as(attn)
            else:
                attn.masked_fill_(mask == 0, -1e30).type_as(attn)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
