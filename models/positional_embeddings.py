




import math
import torch
import torch.nn as nn
import torch.autograd.variable as Variable


def init_weights(emb, padding_idx=None, init=0.1, decay=False, eps=10):
    emb.weight.data.uniform_(-init, init)
    if decay:
        n_vocab, dim = emb.weight.data.size()
        nums = torch.exp(-torch.arange(n_vocab).float() / eps)\
                   .repeat(dim, 1).permute(1, 0) + 1
        nums = nums.to(emb.weight.data.device)
        emb.weight.data *= nums
    if padding_idx is not None:
        emb.weight.data[padding_idx] = 0
    return emb


class ReversePositionalEncoding(nn.Module):
    """Implement the PE function.
    from http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    def __init__(self, d_model, dropout, max_len=5000, attn_direct=False,
                 init_decay=False):
        super(ReversePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # register pe as nn.Embedding
        self.pe = nn.Embedding(max_len, d_model, padding_idx=0)
        self.pe = init_weights(self.pe, padding_idx=0, decay=init_decay)
        self.attn_direct = attn_direct

    def forward(self, x, lengths=None, time_seq=None):
        if not self.attn_direct:
            assert lengths is not None
            n_batch, n_seq, d_emb = x.size()
            pos = torch.arange(n_seq, 0, step=-1).unsqueeze(0).repeat(
                n_batch, 1).to(x.device)
            lengths = lengths.repeat(n_seq,1).permute(1,0).to(x.device)
            pos = pos + lengths - n_seq
            pos = torch.max(pos, torch.zeros(1).long().to(x.device))

            x = x + self.pe(pos)

        else:
            assert time_seq is not None
            x = x + self.pe(time_seq)

        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """Implement the PE function.
    from http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    def __init__(self, d_model, dropout, max_len=5000, sparse_sinusoid=False,
                 use_sinusoid=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.use_sinusoid = use_sinusoid
        self.sparse_sinusoid = sparse_sinusoid

        if use_sinusoid:
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            if sparse_sinusoid:
                # register pe as nn.Embedding
                self.pe = nn.Embedding(max_len, d_model)
                self.pe.load_state_dict({'weight': pe})
                self.pe.weight.requires_grad = False

            else:
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

        else:
            # simple positional embedding
            self.pe = nn.Embedding(max_len, d_model, padding_idx=0)
            self.pe = init_weights(self.pe, padding_idx=0)

    def forward(self, x, time_seq=None, lengths=None):

        if self.use_sinusoid:
            if self.sparse_sinusoid:
                assert time_seq is not None
                pe = self.pe(time_seq)
            else:
                pe_len = x.size(1)
                pe = Variable(self.pe[:, :pe_len], requires_grad=False)

                if x.size(0) != pe.size(0) and len(x.size()) > len(pe.size()):
                    pe = pe.unsqueeze(2)
                    pe = pe.expand_as(x)

        else:
            assert lengths is not None
            n_batch, n_seq, d_emb = x.size()
            pos = torch.arange(1, n_seq + 1).unsqueeze(0).repeat(
                n_batch, 1).to(x.device)

            lengths = lengths.repeat(n_seq, 1).permute(1, 0).to(x.device)
            pos = (pos <= lengths).long() * pos

            pe = self.pe(pos)

        x = x + pe

        return self.dropout(x)

class BackPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super(BackPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        self.r_max, self.r_min = 1.0, 0.01

    def scale(self, vec, r_max, r_min):
        d_max, d_min = max(vec), min(vec)
        scale_factor = (r_max - r_min) / (d_max - d_min)
        return scale_factor * vec

    def forward(self, x, **kwargs):

        max_len = min(self.max_len, x.size(1))
        pad_len = x.size(1) - self.max_len
        # compute backward positional encodings
        # bpe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # bpe[:, 0::2] = torch.exp(position*gamma)*gamma
        bpe = torch.zeros(max_len, self.d_model)
        bpe[:, :] = self.scale(position, self.r_max, self.r_min)

        if x.is_cuda:
            bpe = bpe.to(x.device)

        if pad_len > 0:
            n_batch, seq_len, _ = x.size()
            pad = torch.zeros(pad_len, self.d_model)

            if x.is_cuda:
                pad = pad.to(x.device)
            bpe = torch.cat((pad, bpe))

        bpe = bpe.unsqueeze(0)
        x = x + Variable(bpe, requires_grad=False)
        return self.dropout(x)


class PositionalEncodingConcat(PositionalEncoding):
    """Implement the PE function.
    from http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    def __init__(self, d_model, d_posemb, dropout, max_len=5000):
        super(PositionalEncodingConcat, self).__init__(
            d_model=d_posemb, dropout=dropout, max_len=max_len
        )
        self.W_proj = nn.Linear(d_posemb + d_model, d_model)

    def forward(self, x):
        pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)

        if x.size(0) > pe.size(0) and pe.size(0) == 1:
            pe = pe.expand((x.size(0), pe.size(1), pe.size(2)))

        x_cat = torch.cat((x, pe), 2)
        x = self.W_proj(x_cat)
        return self.dropout(x)
