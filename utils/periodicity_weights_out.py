import numpy as np
import torch

dic_path = 'trained/fml_2019_11_16_02_31_04_LSTM-attn_w12_s4'
dic_file = 'LSTM-attn_w12_s4_event_dic_target_id.npy'

model_path = 'trained/fml_2019_11_15_13_39_32_LSTM-PP-vbv-v12_w12_s4/'
model_file = 'LSTM-PP-vbv-v12_w12_s4_final.model'

model = torch.load('{}/{}'.format(model_path, model_file))
dic = np.load('{}/{}'.format(dic_path, dic_file)).item()

len(dic)
pp_weight = model['pp_weight'].cpu().tolist()
len(pp_weight)

pp_weight = {(i, v) for i, v in enumerate(pp_weight)}

# sort weights
pp_weight = sorted(pp_weight, key=lambda x: x[1])

for idx, val in pp_weight:
    print('{:0.4f} , {}'.format(val, dic[idx+1]['label']))
