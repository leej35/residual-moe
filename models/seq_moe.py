import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru_predictor import GRUPredictor


class SeqMoE(nn.Module):
    def __init__(
        self, input_size, embed_size, output_size, num_experts, hidden_size,
        noisy_gating=True, k=4, dropout=0.2, gate_type='mlp',
        residual=False, base_gru=None, use_zero_expert=False, feed_error=False,
        incl_base_to_expert=False,
    ):
        super(SeqMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.feed_error = feed_error
        self.incl_base_to_expert = incl_base_to_expert
        
        self.base_gru = base_gru    
   
        self.embed_layer = nn.Linear(input_size, embed_size, bias=False)
        if self.base_gru:
            # lock base gru parameters
            self.embed_layer.weight = base_gru.embed_input.weight
            self.base_gru.embed_input.weight.requires_grad = False
            for param in self.base_gru.rnn.parameters():
                param.requires_grad = False
            for param in self.base_gru.fc_out.parameters():
                param.requires_grad = False

        # instantiate experts
        # MLP(self.input_size, self.output_size, self.hidden_size)
        inp_dim_expert = embed_size
        if self.feed_error:
            inp_dim_expert += output_size

        self.experts = nn.ModuleList(
            [GRUPredictor(inp_dim_expert, self.hidden_size, self.output_size, dropout)
                for i in range(self.num_experts)])

        num_experts = self.num_experts
        if use_zero_expert:
            num_experts += 1
            self.experts.append(ZeroModule(self.output_size))

        if self.incl_base_to_expert:
            num_experts += 1

        # curren time step embed => number of experts
        if gate_type == 'mlp':
            self.gate = nn.Linear(embed_size, num_experts)
        elif gate_type == 'gru':
            self.gate = GRUPredictor(
                embed_size, self.hidden_size, num_experts, dropout)

    def forward(self, inp, inp_lengths, trg):

        device = inp.device
        batch_size = inp.size(0)

        inp_embed = self.embed_layer(inp)

        gate_val = self.gate(inp_embed)
        gate_val = F.softmax(gate_val, dim=2)
        # gate_val: n_batch x n_seq x n_gate

        n_batch, n_seq, _ = inp.shape

        if self.base_gru:
            base_init_hidden = GRUPredictor.init_hidden(
                batch_size, self.base_gru.hidden_dim
            ).to(device)

            preds, _, _ = self.base_gru(
                inp, inp_lengths, base_init_hidden
            )
            preds = F.sigmoid(preds)

        else:
            preds = torch.zeros(n_batch, n_seq, self.output_size).to(device)

        if self.incl_base_to_expert and self.base_gru:
            gate_exp_val = gate_val[:, :, -1].unsqueeze(2)
            # gate_exp_val: n_batch x n_seq x 1
            
            preds = preds * gate_exp_val

        if self.feed_error:
            if self.base_gru == None:
                raise RuntimeError("should have base_gru to run feed_error")
            errors = F.l1_loss(preds, trg, reduction='none')

            # shift timesteps +1, and add zero padding for the first one
            errors = errors[:, :-1, :]
            zero_pad = torch.zeros(batch_size, 1, self.output_size)
            zero_pad = zero_pad.to(errors.device)
            prev_errors = torch.cat((zero_pad, errors), dim=1)

        for exp_idx, expert in enumerate(self.experts):
            
            if self.feed_error:
                inp_expert = torch.cat((inp_embed, prev_errors), dim=2)
            else:
                inp_expert = inp_embed

            pred_exp = expert(inp_expert)
            # pred_exp: n_batch x n_seq x n_events
            
            gate_exp_val = gate_val[:, :, exp_idx].unsqueeze(2)
            # gate_exp_val: n_batch x n_seq x 1
            
            preds = preds + gate_exp_val * pred_exp
        
        preds = preds.clamp(0, 1)
        return preds, gate_val
        
    def init_hidden(self, batch_size):
        return GRUPredictor.init_hidden(batch_size, self.hidden_size)
        
class ZeroModule(nn.Module):
    def __init__(self, output_size):
        super(ZeroModule, self).__init__()
        self.output_size = output_size

    def forward(self, inp):
        batch_size, n_seq, _ = inp.size()
        zero_tensor = torch.zeros(batch_size, n_seq, self.output_size)
        return zero_tensor.to(inp.device)
