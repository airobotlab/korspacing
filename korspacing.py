# -*- coding: utf-8 -*-
# import
import os
import re
import argparse
import json
from pprint import pprint
import torch
import torch.nn.functional as F
import numpy as np
from utils import CharLevelLSTM, pre_processing, encoding_and_padding, load_vocab, pad_sequences, make_pred_sents, inference

root_folder = './resources'
## config
rules_path = os.path.join(root_folder, 'rules.json')
vocab_path = os.path.join(root_folder, 'w2idx.dic')
model_path = os.path.join(root_folder, 'space_model.pth')

# model_path = pkg_resources.resource_filename(
#     'korspacing', os.path.join('resources', 'space_model.pth'))
# vocab_path = pkg_resources.resource_filename(
#     'korspacing', os.path.join('resources', 'dicts', 'c2v.dic'))

max_seq_len = 200

# 하이퍼파라미터 설정
config_model_vocab_size = 1994
config_model_embed_dim = 256
config_model_num_filters = 128
config_model_filter_sizes = [2, 3, 4, 5]  # 다양한 크기의 필터 사용
config_model_hidden_dim = 256
config_model_output_dim = 2  # binary
config_model_dropout = 0.3
config_model_num_layers = 3
config_model_pad_token=1993  # padding 무시하고 loss를 구하기 위해

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharLevelLSTM(config_model_vocab_size, config_model_embed_dim, config_model_hidden_dim, config_model_output_dim, config_model_num_layers, config_model_dropout)
model.load_state_dict(torch.load(model_path));  # Best 상태로 모델을 로드
model.eval()
model.to(device);

# 사전 파일 로딩
w2idx, idx2w = load_vocab(vocab_path)  # 1994, w2idx: '엠': 1352


## KorSpacing class
class KorSpacing:
    '''
    run korean word spacing by torch
    '''
    def __init__(self, rules=True, verbose=False):
        self.w2idx = w2idx
        self.max_seq_len = max_seq_len
        self.device = device
        self.verbose = verbose        
        self.model = model
        
        # rules가 dictionary면 바로 읽고, 경로면 dictionary를 읽어온다
        if type(rules)==dict:
            self.rules = rules
        elif rules==True:
            with open(rules_path, 'r', encoding='UTF-8') as f_read:
                self.rules = json.load(f_read)
                
        print('## rules ##')
        pprint(self.rules)

    def __call__(self, input_txt):
        
        # 띄어쓰기 제거
        input_txt_nospace = input_txt.replace(" ", "")
        
        if len(input_txt_nospace) > self.max_seq_len:
            splitted_sent = [input_txt_nospace[:self.max_len]]
        else:
            splitted_sent = [input_txt_nospace]
        
        input_txt_processed = pre_processing(splitted_sent)  # ['«아버지가^방에^들어^가^신다»']
        input_txt_processed_no_space = [tmp.replace('^', '') for tmp in input_txt_processed]  # ['«아버지가방에들어가신다»']

        input_txt_processed_indice = encoding_and_padding(word2idx_dic=self.w2idx,
                                      sequences=input_txt_processed_no_space,
                                      maxlen=self.max_seq_len,
                                      padding='post',
                                      truncating='post')  # [[   3,   26,  188,   12,   11,  118,    7,   29,   20,   11,   70, 2,   4, 1993, 1993,
        input_ids = torch.tensor(input_txt_processed_indice).to(device)
        
        # run model
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Apply softmax to outputs to get probabilities
            probabilities = F.softmax(outputs, dim=-1)  # (batch_size, seq_length, num_classes)
            # Convert outputs to predicted labels by taking the argmax
            predicted_labels = torch.argmax(probabilities, dim=2)  # (batch_size, seq_length)
            predictions = predicted_labels.cpu().numpy().squeeze()

            pad_index_from = np.where(input_txt_processed_indice[0] == 1993)[0][0]
            predictions = predictions[:pad_index_from].tolist()

            result_spacing = make_pred_sents(list(input_txt_processed_no_space[0]), predictions)
            
            if self.verbose:
                print(f'{input_txt}\n=>{input_txt_nospace}\n=>{input_txt_processed_no_space}\n=>{result_spacing}\n=>{predictions}')        

        # rule 적용
        if len(self.rules)>0:
#             print('# rule #')
            for word, rgx in rules.items():
                result_spacing = result_spacing.replace(word, rgx)
        
        return result_spacing.strip()

    

from korspacing import KorSpacing

rules = {
    '아버지 가방에': '아버지가 방에', 
    '아 버지가방': '아버지가 방',     
}

# spacing = KorSpacing(rules=rules)
spacing = KorSpacing()
result = spacing('아버지가방에들어가신다')
print(result)

