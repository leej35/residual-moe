import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence, pad_sequence
)

from utils.trainer_utils import get_hash


class GRUPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout):
        super(GRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embed_dim, hidden_dim, dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedded_inp, lengths):

        batch_size = embedded_inp.size(0)
        device = embedded_inp.device
        hidden = self.init_hidden(batch_size, self.hidden_dim).to(device)

        hidden, _ = self.gru(embedded_inp, hidden)

        prediction = self.proj_out(hidden)
        # prediction = F.sigmoid(prediction)
        return prediction

    @staticmethod
    def init_hidden(batch_size, hidden_dim):
        init = 0.1
        h0 = torch.randn(batch_size, hidden_dim)
        h0.data.uniform_(-init, init)
        return h0.unsqueeze(0)


class SequentialTwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SequentialTwoLayerNet, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        hidden = GRUPredictor.init_hidden(
            x.size(0), self.hidden_dim).to(x.device)

        h, _ = self.gru1(x, hidden)
        h = F.leaky_relu(h)
        h, _ = self.gru2(h)
        o = self.fc(h)
        return o


class SubGroupGRUs(nn.Module):
    def __init__(
        self, 
        input_dim,
        embed_dim,
        hidden_dim, 
        output_dim,
        num_group,
        base_gru,
        svd,
        kmeans,
        dropout,
        way_output,
        group_info,
        sg_init_gru_from_base=False,
        threshold_errors=False,
        threshold_errors_value=0.5,
        concat_base_hidden=False,
        add_base_hidden=False,
        adapt_on_error=False,
        sg_loss_func=F.binary_cross_entropy,
        topk=0,
    ):
        """
        adapt_on_error: 
            - True: model try to learn amount to adapt (difference between
            true target and base model's prediction) and the adaptation amount
            will be added to the prediction from base model.
        """
        super(SubGroupGRUs, self).__init__()
        self.num_group = num_group
        self.svd = svd
        self.kmeans = kmeans
        self.way_output = way_output
        self.base_gru = base_gru
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold_errors = threshold_errors
        self.threshold_errors_value = threshold_errors_value

        self.input_emb = nn.Linear(input_dim, embed_dim, bias=False)
        self.input_emb.weight = base_gru.embed_input.weight

        self.sub_grus = nn.ModuleList(
            [GRUPredictor(embed_dim, hidden_dim, output_dim, dropout)
             for _ in range(num_group)]
        )
        
        self.base_hidden_dim = base_gru.rnn.hidden_size
        if sg_init_gru_from_base and self.base_hidden_dim == hidden_dim:
            for sub_gru in self.sub_grus:
                sub_gru.gru.weight_ih_l0.data = base_gru.rnn.weight_ih_l0.data
                sub_gru.gru.weight_hh_l0.data = base_gru.rnn.weight_hh_l0.data
                sub_gru.gru.bias_ih_l0.data = base_gru.rnn.bias_ih_l0.data
                sub_gru.gru.bias_hh_l0.data = base_gru.rnn.bias_hh_l0.data
                sub_gru.proj_out.bias.data = base_gru.fc_out.bias.data
                sub_gru.proj_out.weight.data = base_gru.fc_out.weight.data
        self.group_info = group_info

        self.add_base_hidden = add_base_hidden
        self.concat_base_hidden = concat_base_hidden
        if self.concat_base_hidden:

            if way_output == "W2":
                # W2: concatenate all subGRUs and let SGD it figure out best
                n_chunks = (num_group + 1)
                
            elif way_output == "W1":
                # W1: we only use top-1 subGRU
                n_chunks = 2
            else:
                n_chunks = 1

            self.concat_fc = nn.Linear(output_dim * n_chunks, output_dim)
            self.concat_fc.weight.data *= torch.cat(
                [torch.eye(output_dim, output_dim)] * n_chunks, dim=1)
        
        if way_output == "W2":
            # MOE
            self.seq_experts = SequentialTwoLayerNet(
                input_dim=output_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )

        if way_output == "W3":
            # MultiHead Attention
            # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            self.attn = nn.MultiheadAttention(
                embed_dim=input_dim, 
                num_heads=1, 
                dropout=dropout, 
                bias=True,
            )
        
        self.topk = topk
        self.adapt_on_error = adapt_on_error
        self.sg_loss_func = sg_loss_func

    def forward(self, inp_seq, inp_lengths, trg_seq, mode):
        if self.adapt_on_error:
            return self.forward__V2(inp_seq, inp_lengths, trg_seq, mode)
        else:
            return self.forward_V1(inp_seq, inp_lengths, trg_seq, mode)

    def forward__V2(self, inp_seq, inp_lengths, trg_seq, mode='train'):
        """
        forward V2: combine train and test code separation
        """

        device = inp_seq.device
        batch_size, seq_len, _ = inp_seq.size()
        train_loss = 0
        pred = None

        base_init_hidden = GRUPredictor.init_hidden(
            batch_size, self.base_gru.hidden_dim
        ).to(device)

        pred_basem_seq, _, _ = self.base_gru(
            inp_seq, inp_lengths, base_init_hidden
        )
        pred_basem_seq = F.sigmoid(pred_basem_seq)

        seq_errors = self.sg_loss_func(
            pred_basem_seq, trg_seq, reduction='none')
            
        
        if mode == 'pretrain':
            # then we already have group information
            # -- but we still need to query to hash and match with each label info..
            #
            # retrieve group info
            group_batch = [
                self.group_info[get_hash(inst_seq.cpu().detach().numpy())]
                for inst_seq in inp_seq
            ]

            # Feed input to modules
            for group in range(self.num_group):
                g_mask, g_lengths = self.get_group_mask_lengths(
                    group_batch, inp_lengths, group)

                if sum(g_lengths) == 0:
                    continue

                g_mask = g_mask.to(device)
                g_inp = inp_seq[g_mask != 0, :, :]

                #NOTE: target is error at next step.
                g_trg = seq_errors[g_mask != 0, :, :]
                g_emb = self.input_emb(g_inp)
                g_pred = self.sub_grus[group](g_emb, g_lengths)
                g_loss = self.sg_loss_func(g_pred, g_trg)
                train_loss = train_loss + g_loss

        elif mode in ['train', 'test']:
            # we need to inference group info based on "Recent" error

            emb_seq = self.input_emb(inp_seq)
            if self.way_output in ["W1", "W0"]:
                """
                as dictionary
                """
                subgroups_pred = {
                    group: self.sub_grus[group](emb_seq, inp_lengths)
                    for group in range(self.num_group)
                }
            elif self.way_output in ["W2", "W3"]:
                """
                as tensor
                """
                subgroups_pred = [
                    self.sub_grus[group](emb_seq, inp_lengths)
                    for group in range(self.num_group)
                ]
                subgroups_pred = torch.stack(subgroups_pred, dim=0)

            # Now, how to merge?

            pred = []
            for t in range(seq_len):
                pred_step = pred_basem_seq[:, t, :]

                if t > 0:
                    adapt_step = torch.zeros(
                        batch_size, self.output_dim).to(device)

                    # from the 2nd timestep, we have access to previous history
                    # this is why we use seq_errors[t-1]
                    prev_step_errors = seq_errors[:,
                                                  t - 1, :].cpu().detach().numpy()

                    if self.threshold_errors:
                        prev_step_errors = prev_step_errors < self.threshold_errors_value

                    svd_embedding = self.svd.transform(prev_step_errors)

                    if self.way_output == "W0":
                        # kmeans.transform returns distance to the cluster centers
                        # this uses euclidean distance
                        prev_step_group_dist = self.kmeans.transform(
                            svd_embedding)

                        # prev_step_group_dist : n_batch x num_clusters
                        prev_step_group_dist = torch.FloatTensor(
                            prev_step_group_dist).to(device)

                        # TODO: sum-to-one through softmax should not be enough
                        #       to get the correct combined output.
                        #       we might need certain scaling factor.
                        (prev_step_group_dist.pow_(2) * -1).exp_()

                        if self.topk > 0:    
                            prev_step_group_dist = get_masked_topk(
                                prev_step_group_dist, self.topk,
                            )

                        convex_combi_weight = F.softmax(
                            prev_step_group_dist, dim=1)

                        for group in range(self.num_group):
                            g_weight = convex_combi_weight[:, group]
                            g_pred = subgroups_pred[group][:, t, :]
                            adapt_step += g_weight.unsqueeze(1) * g_pred

                    elif self.way_output == "W1":
                        """
                        use top-1 kmeans distance group's prediction
                        """
                        group_info = self.kmeans.predict(svd_embedding)
                        pred_batch = []
                        for batch_group in list(group_info):
                            pred_batch.append(
                                subgroups_pred[batch_group][:, t, :]
                            )
                        adapt_step = torch.stack(pred_batch, dim=0)
                    
                    elif self.way_output == "W2":
                        """
                        Mixture of Experts
                        """

                        # group x batch x time x event_dim
                        experts_inp = subgroups_pred[:, :, t, :]
                        self.seq_experts(experts_inp)

                        # self.topk
                    elif self.way_output == "W3":
                        """
                        Attention Mechanism
                        """
                        pass                        


                    pred_step = pred_step + adapt_step
                    pred_step.clamp_(0, 1)

                else:
                    # first timestep: we don't have target info yet
                    #    => we must use base gru prediction which does not need
                    pass

                pred.append(pred_step)


            pred = torch.stack(pred, dim=1)

        return train_loss, pred

    def forward_V1(self, inp_seq, inp_lengths, trg_seq, mode='train'):
        
        device = inp_seq.device
        batch_size, seq_len, _ = inp_seq.size()
        train_loss = 0
        pred = None

        if mode == 'train': 
            # then we already have group information 
            # -- but we still need to query to hash and match with each label info..
            
            # retrieve group membership info (one id for each batch element)
            group_batch = [
                self.group_info[get_hash(inst_seq.cpu().detach().numpy())] \
                    for inst_seq in inp_seq
            ]

            # Feed input to modules
            for group in range(self.num_group):
                g_mask, g_lengths = self.get_group_mask_lengths(
                    group_batch, inp_lengths, group)
                
                if sum(g_lengths) == 0:
                    continue

                g_mask = g_mask.to(device)
                g_inp = inp_seq[g_mask != 0, :, :]
                g_trg = trg_seq[g_mask != 0, :, :]
                g_emb = self.input_emb(g_inp)
                g_pred = self.sub_grus[group](g_emb, g_lengths)
                g_pred = F.sigmoid(g_pred)
                g_loss = F.binary_cross_entropy(g_pred, g_trg)
                train_loss = train_loss + g_loss

        if mode == 'test':
            # we need to inference group info based on "Recent" error

            base_init_hidden = GRUPredictor.init_hidden(
                batch_size, self.base_hidden_dim
            ).to(device)

            pred_basem_seq, _, _ = self.base_gru(
                inp_seq, inp_lengths, base_init_hidden
            )
            pred_basem_seq = F.sigmoid(pred_basem_seq)
            seq_errors = F.binary_cross_entropy(
                pred_basem_seq, trg_seq, reduction='none')

            # subgroups' prediction
            emb_seq = self.input_emb(inp_seq)
            subgroups_pred = {
                group: self.sub_grus[group](emb_seq, inp_lengths) \
                for group in range(self.num_group)
            }

            pred = []
            for t in range(seq_len):
                adapt_step = torch.zeros(batch_size, self.output_dim).to(device)

                if t > 0:
                    # from the 2nd timestep, we have access to previous history
                    # this is why we use seq_errors[t-1]
                    prev_step_errors = seq_errors[:, t - 1, :].cpu().detach().numpy()

                    if self.threshold_errors:
                        prev_step_errors = prev_step_errors < self.threshold_errors_value

                    svd_embedding = self.svd.transform(prev_step_errors)

                    if self.way_output == "W0":
                        """
                        W0: convex combination for all groups based on its distance
                        """

                        # kmeans.transform returns distance to the cluster centers
                        # this uses euclidean distance
                        prev_step_group_dist = self.kmeans.transform(svd_embedding)

                        # prev_step_group_dist : n_batch x num_clusters
                        prev_step_group_dist = torch.FloatTensor(
                            prev_step_group_dist).to(device)

                        # TODO: sum-to-one through softmax should not be enough
                        #       to get the correct combined output.
                        #       we might need certain scaling factor. 
                        (prev_step_group_dist.pow_(2) * -1).exp_()
                        convex_combi_weight = F.softmax(prev_step_group_dist, dim=1)
                        
                        for group in range(self.num_group):
                            g_weight = convex_combi_weight[:, group]
                            g_pred = subgroups_pred[group][:, t, :]
                            g_pred = F.sigmoid(g_pred)
                            adapt_step += g_weight.unsqueeze(1) * g_pred

                    elif self.way_output == "W1":
                        """
                        W1: top-1 group's info is used.
                        """

                        # group_info: each batch element's (=seq) membership info
                        group_info = self.kmeans.predict(svd_embedding)
                        pred_batch = []
                        for batch_group in list(group_info):
                            g_pred = subgroups_pred[batch_group][:, t, :]
                            g_pred = F.sigmoid(g_pred)
                            pred_batch.append(g_pred)

                        adapt_step = torch.stack(pred_batch, dim=0)
                        
                        if self.add_base_hidden:
                            pred_batch += pred_basem_seq[:, t, :]

                    else:
                        raise NotImplementedError

                else:
                    # first timestep: we don't have target info yet
                    #    => we must use base gru prediction which does not need
                    adapt_step = pred_basem_seq[:, t, :]
                pred.append(adapt_step)

            pred = torch.stack(pred, dim=1)

        return train_loss, pred

    def get_group_mask_lengths(self, group_batch, inp_lengths, group):
        """Compute group masks where each has size of (n_batch)
        mask has 1 where group matches 
        Args:
            inp_seq (torch.FloatTensor): input tensor, n_batch x n_seq x n_event
            group_batch (list[int]): current batch's group membership info
        """
        g_mask = (torch.FloatTensor(group_batch) == group).float()
        g_lengths = inp_lengths * g_mask.long()
        g_lengths = g_lengths[g_lengths != 0].tolist()

        return g_mask, g_lengths

    def init_hidden(self, batch_size):
        init = 0.1
        hidden_dim = self.hidden_dim
        h0 = torch.randn(batch_size, hidden_dim)
        h0.data.uniform_(-init, init)
        return h0


def get_masked_topk(mat, topk, masked_val=float('-inf')):
    topk_vals, topk_idxs = mat.topk(topk, dim=1)
    res = torch.ones(mat.size()).to(mat.device)
    res = res * masked_val
    res = res.scatter(1, topk_idxs, topk_vals)
    return res
