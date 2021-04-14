''' Define the Layers '''
import torch.nn as nn
import torch
import torch.utils
import torch.utils.checkpoint


from models.transformer.SubLayers import MultiHeadAttention, MultiHeadAttention4d, PositionwiseFeedForward
# from SubLayers import MultiHeadAttention, MultiHeadAttention4d, PositionwiseFeedForward

"""
Code based on https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, tmp_input=None, slf_attn_mask=None):
        '''
        enc_input : (n_batch x [seq_len] x d_word_vec)
        tmp_input : (n_batch x [seq_len] x d_clock)
        slf_attn_mask : (n_batch x seq_len x seq_len)
        '''
        
        # enc_output, enc_slf_attn = torch.utils.checkpoint.checkpoint(enc_input, enc_input, enc_input, tmp_input, tmp_input, slf_attn_mask)
        enc_output, enc_slf_attn = self.slf_attn(
            q=enc_input, k=enc_input, v=enc_input, q_time=tmp_input, k_time=tmp_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, tmp_input, enc_slf_attn


class PredictorLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(PredictorLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_output, tmp_output, tmp_input=None, subseq_io_mask=None):
        
        pred_output, pred_slf_attn = self.slf_attn(
            q=tmp_output, k=enc_output, v=enc_output, q_time=tmp_input, k_time=tmp_input, mask=subseq_io_mask)
        pred_output = self.pos_ffn(pred_output)

        return pred_output, pred_slf_attn
