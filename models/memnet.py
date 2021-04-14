import math
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y/%m/%d %I:%M:%S %p')

logger.setLevel(logging.DEBUG)



class MemNetE2E(nn.Module):
    """
    E2E : (key) PrevError to (value) NextError
    """

    def __init__(self, key_size, value_size, num_slots=None, policy="age", is_unbounded=False):
        super(MemNetE2E, self).__init__()
        
        self.key_size = key_size
        self.value_size = value_size
        self.is_unbounded = is_unbounded

        self.is_written = False
        self.policy = policy
        self.reset_mem()
        self.num_slots = num_slots

    def reset_mem(self):

        if self.is_unbounded:
            self.num_slots = 0
            self.keys = nn.Parameter(torch.zeros(0, self.key_size))
            self.values = nn.Parameter(torch.zeros(0, self.value_size))
            self.errors = nn.Parameter(torch.zeros(0))
            self.ages = nn.Parameter(torch.zeros(0))
        else:
            self.keys = nn.Parameter(torch.rand(self.num_slots, self.key_size))
            self.values = nn.Parameter(torch.rand(self.num_slots, self.value_size))
            self.ages = nn.Parameter(torch.zeros(self.num_slots))
            self.errors = nn.Parameter(torch.zeros(self.num_slots))

    def read_mem(self, query, read_mode='nn1', as_batch=True, content_avg=True, width=0.2, shrink_event_dim=False):
        
        n_batch = query.size(0)
        error_slot, similarity_slot = None, None

        tensor_size = (n_batch, self.num_slots, self.key_size)
        query = query.unsqueeze(1)
        key_batch = self.keys.unsqueeze(0).expand(tensor_size)

        if read_mode.startswith('nn') or read_mode == 'softmax':
            key_by_query = torch.cosine_similarity(
                query.expand(tensor_size), key_batch, dim=2)

        if read_mode.startswith('nn'):
            topk = int(read_mode.lstrip('nn'))
            # read_pos = torch.argmax(key_by_query, dim=1)
            val, read_pos = torch.topk(key_by_query, dim=1, k=topk)

            with torch.no_grad():
                if not self.is_unbounded:
                    self.ages[read_pos] += 1
            contents = self.values[read_pos]
            error_slot = self.errors[read_pos]
            similarity_slot, _ = key_by_query.max(1)

            if content_avg:
                contents = contents.mean(1)
                error_slot = error_slot.mean(1)

        elif read_mode == 'softmax':
            addresses = torch.softmax(key_by_query, dim=1).unsqueeze(2)
            contents = torch.sum(addresses * self.values, dim=1)

        elif read_mode in ['gaussian_kernel', 'hamming_distance']:
            if read_mode == 'gaussian_kernel':
                weights = gaussian_kernel(
                    # query if shrink_event_dim else query.expand(tensor_size),
                    query.expand(tensor_size),
                    key_batch,
                    width)

            elif read_mode == 'hamming_distance':
                weights = hamming_distance(
                    query.expand(tensor_size),
                    key_batch
                )
            else:
                raise NotImplementedError
            

            if shrink_event_dim:

                weights = weights.mean(2).unsqueeze(2)
                values = self.values.unsqueeze(0)
                weights = weights / weights.sum(1).unsqueeze(2)
                contents = weights * values
                contents = torch.sum(contents, dim=1)

                # previous way
                # weights.squeeze_(1).unsqueeze_(-1)

                # weights = weights / weights.sum(1).unsqueeze(1)
                # values = self.values.unsqueeze(0)
                # contents = weights * values
                # contents = torch.sum(contents, dim=1)
            else:
                weights = weights / weights.sum(1).unsqueeze(1)
                weights.mul_(self.values)
                contents = torch.sum(weights, dim=1)

        else:
            raise NotImplementedError

        contents = contents.detach()
        if error_slot is not None:
            error_slot = error_slot.detach()
        if similarity_slot is not None:
            similarity_slot = similarity_slot.detach()

        return contents, error_slot, similarity_slot

    def write_mem(self, key, value, error=None):
        """
        key: hidden states and error_prev
        value: error_cur

        error_prev : n_batch x input_size
        """
        n_batch = key.size(0)
            
        if self.is_unbounded:
            self.keys = nn.Parameter(torch.cat((self.keys, key), dim=0))
            self.values = nn.Parameter(torch.cat((self.values, value), dim=0))
            self.errors = nn.Parameter(torch.cat((self.errors, error.mean(-1))))
            self.num_slots = self.keys.size(0)
        else:
            for b_idx in range(n_batch):

                if self.policy == 'age':
                    # write policy: write at oldest position
                    write_pos = torch.argmin(self.ages)
                    self.ages[write_pos] = 1  # reset after writing

                elif self.policy == 'error':
                    # write policy: write at smallest error position
                    # (reasoning:) the more error population model has, the more worth target info  
                    write_pos = torch.argmin(self.errors)
                    self.errors[write_pos] = error[b_idx].mean()

                self.values[write_pos] = value[b_idx]
                self.keys[write_pos] = key[b_idx]

        return None

    def log_mem_to_file(self, path):
        os.system(f"mkdir -p {path}")
        logger.debug(f"log mem contents into files in {path}")
        with open(f"{path}/keys.npy", 'wb') as f:
            np.save(f, self.keys.cpu().detach().numpy())
        with open(f"{path}/values.npy", 'wb') as f:
            np.save(f, self.values.cpu().detach().numpy())
        with open(f"{path}/ages.npy", 'wb') as f:
            np.save(f, self.ages.cpu().detach().numpy())
        with open(f"{path}/errors.npy", 'wb') as f:
            np.save(f, self.errors.cpu().detach().numpy())


def hamming_distance(a, b):
    distance = (a == b)
    return distance

def gaussian_kernel(a, b, width):
    # if shrink_event_dim:
    #     distance = torch.cdist(a, b, p=2.0)
    # else:
    distance = torch.abs(a - b)
    del a, b
    const = 1 / math.sqrt(2 * math.pi * (width ** 2))
    # result = const * torch.exp(- distance / (2 * (width ** 2)))
    return distance.pow_(2).div_((2 * (width ** 2))).mul_(-1).exp_().mul_(const)


class LTAMControllerV1(nn.Module):
    """
    V1: input: hidden state only
    """
    def __init__(self, input_size, output_size, hidden_size, act_func="gelu"):
        super(LTAMControllerV1, self).__init__()
        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.mlp4 = nn.Linear(hidden_size, output_size)
        
        if act_func == "gelu":
            self.act_func = nn.GELU()
        elif act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == "tanh":
            self.act_func = nn.Tanh()
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
    def forward(self, hidden_state):
        return self.sigmoid(self.mlp4(self.act_func(self.mlp3(self.act_func(
            self.mlp2(self.act_func(self.mlp1(hidden_state))))))))


class LTAMControllerV2(nn.Module):
    """
    V2: input: hidden state + reading of the memory
    """

    def __init__(self, hidden_sizes, p_dropout=0.5, act_func="gelu", ):
        super(LTAMControllerV2, self).__init__()
        
        if act_func == "gelu":
            self.act_func = nn.GELU()
        elif act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == "tanh":
            self.act_func = nn.Tanh()
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p_dropout)
        self.MLP = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.MLP.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))

    def forward(self, read_input):
        for mlp in self.MLP:
            read_input = self.act_func(self.dropout(mlp(read_input)))
        return self.sigmoid(read_input)


class LTAMControllerV3(nn.Module):
    """
    V3: input: hidden state + reading of the memory
        Use RNN.
    """

    def __init__(self, hidden_sizes, p_dropout=0.5, act_func="gelu", ):
        super(LTAMControllerV3, self).__init__()

        if act_func == "gelu":
            self.act_func = nn.GELU()
        elif act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == "tanh":
            self.act_func = nn.Tanh()
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p_dropout)
        self.MLP = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.MLP.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))

    def forward(self, hidden_state, predicted_read):
        read_input = torch.cat((hidden_state, predicted_read), dim=1)
        for mlp in self.MLP:
            read_input = self.act_func(self.dropout(mlp(read_input)))
        return self.sigmoid(read_input)


class ContextTargetMLP(nn.Module):
    """
    """

    def __init__(self, hidden_sizes, p_dropout=0.5, act_func="gelu", ):
        super(ContextTargetMLP, self).__init__()

        if act_func == "gelu":
            self.act_func = nn.GELU()
        elif act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == "tanh":
            self.act_func = nn.Tanh()
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p_dropout)
        self.MLP = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.MLP.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))

    def forward(self, hidden_state, predicted_read):
        read_input = torch.cat((hidden_state, predicted_read), dim=1)
        for mlp in self.MLP:
            read_input = self.act_func(self.dropout(mlp(read_input)))
        return self.sigmoid(read_input)




class PredErrorModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, act_func="gelu"):
        super(PredErrorModel, self).__init__()
        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.mlp4 = nn.Linear(hidden_size, output_size)
        
        if act_func == "gelu":
            self.act_func = nn.GELU()
        elif act_func == "relu":
            self.act_func = nn.ReLU()
        elif act_func == "tanh":
            self.act_func = nn.Tanh()
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
    def forward(self, hidden_state):
        return self.relu(self.mlp4(self.act_func(self.mlp3(self.act_func(
            self.mlp2(self.act_func(self.mlp1(hidden_state))))))))


class AdaptiveMemProcess(nn.Module):
    def __init__(self, input_size, hidden_size, num_slots, freeze_params=False, freeze_mem=False, input_embed=None, lstm=None, out_proj=None, off_mem=False):
        super(AdaptiveMemProcess, self).__init__()

        self.input_size, self.hidden_size, self.num_slots, self.off_mem \
            = input_size, hidden_size, num_slots, off_mem

        if input_embed is None:
            self.embedding = nn.Linear(input_size, hidden_size)
        else:
            self.embedding = input_embed

        if lstm is None:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.lstm = lstm

        if out_proj is None:
            self.out_proj = nn.Linear(hidden_size, input_size)
        else:
            self.out_proj = out_proj

        self.memnet = MemNetE2E(input_size, input_size,
                                num_slots, policy='error')

        if freeze_params:
            self.freeze_params(self.embedding)
            self.freeze_params(self.lstm)
            # self.freeze_params(self.out_proj)

        if freeze_mem:
            self.freeze_params(self.memnet)

    def freeze_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, mode, inp_seq, trg_seq, hidden_state=None):

        inp_emb = self.embedding(inp_seq)

        preds = []
        error_prev = None
        predicted_error = None

        if hidden_state is None:
            batch_size = inp_seq.size(0)
            h0 = torch.rand(1, batch_size, self.hidden_size)
            c0 = torch.rand(1, batch_size, self.hidden_size)
            hidden_state = (h0, c0)

        lstm_out, _ = self.lstm(inp_emb, hidden_state)

        for t_step in range(inp_seq.size(1)):

            trg_step = trg_seq[:, t_step]
            lstm_out_step = lstm_out[:, t_step]

            pred_step = torch.sigmoid(self.out_proj(lstm_out_step))

            error_cur = trg_step - pred_step

            if error_prev is not None and not self.off_mem:
                if mode == 'train':
                    self.memnet.write_mem(key=error_prev, value=error_cur)
                elif mode == 'test':
                    predicted_error = self.memnet.read_mem(
                        query=error_prev, read_mode='softmax')

            error_prev = error_cur

            if mode == 'test' and predicted_error is not None and not self.off_mem:
                pred_step += predicted_error

            preds.append(pred_step)

        pred_seq = torch.stack(preds, dim=1)
        return pred_seq


def test_AdaptiveMemProcess():
    input_size = output_size = 4
    num_slots = 200
    n_batch = 5
    hidden_size = 20
    seq_len = 10

    epochs = 4000
    learning_rate = 0.001
    print_every = 100


    net = AdaptiveMemProcess(
        input_size, hidden_size, num_slots, freeze_params=False, freeze_mem=False, off_mem=False)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    f_loss = nn.BCELoss()

    seq = (torch.rand(n_batch, seq_len, input_size) > 0.5).float()
    inp_seq, trg_seq = seq[:, :-1], seq[:, 1:]

    for i in range(epochs):
        pred = net("train", inp_seq, trg_seq)

        if i % print_every == 0:
            logger.info(f"train accuracy: {((pred>0.5) == trg_seq).float().mean()}")

        loss = f_loss(pred, trg_seq)
        loss.backward()
        optimizer.step()

        pred = net("test", inp_seq, trg_seq)

        if i % print_every == 0:
            logger.info(f"test accuracy: {((pred>0.5) == trg_seq).float().mean()}")

            logger.info(f"loss: {loss}")
            # logger.info(f"age: {net.memnet.ages}")
            logger.info("*"*10)

def test_MemNetE2E():
    input_size = output_size = 1
    num_slots = 10
    n_batch = 5
    policy = ""

    net = MemNetE2E(input_size, output_size, num_slots, policy)

    error_prev = torch.rand(n_batch, input_size)
    error_cur = torch.rand(n_batch, input_size)

    logger.info(net.write_mem(error_prev, error_cur))
    logger.info(net.ages)

    error_prev = torch.rand(n_batch, input_size)
    error_cur = torch.rand(n_batch, input_size)

    logger.info(net.read_mem(error_prev))
    logger.info(net.ages)

    logger.info(net.write_mem(error_prev, error_cur))
    logger.info(net.ages)

    error_prev = torch.rand(n_batch, input_size)
    error_cur = torch.rand(n_batch, input_size)

    logger.info(net.write_mem(error_prev, error_cur))
    logger.info(net.ages)

    error_prev = torch.rand(n_batch, input_size)
    logger.info(error_prev)
    logger.info(net.read_mem(error_prev))
    logger.info(net.ages)


if __name__ == '__main__':
    test_AdaptiveMemProcess()
