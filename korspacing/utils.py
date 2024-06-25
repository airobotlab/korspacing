# import
import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## functions
# CNN-LSTM 모델 정의
class CharLevelLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes=1, num_layers=1, dropout=0.5):
        super(CharLevelLSTM, self).__init__()
        
        self.num_classes = num_classes
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embed_size)
        x, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_length, hidden_size)
        
        # Apply the fully connected layer to each time step
        x = self.fc(x)  # (batch_size, seq_length, num_classes)
        
        return x.squeeze(-1)  # (batch_size, seq_length)


def pre_processing(setences):
    # 공백은 ^
    char_list = [li.strip().replace(' ', '^') for li in setences]
    # 문장의 시작 포인트 «
    # 문장의 끌 포인트  »
    char_list = ["«" + li + "»" for li in char_list]
    # 문장 -> 문자열
    char_list = [''.join(list(li)) for li in char_list]
    return char_list


def encoding_and_padding(word2idx_dic, sequences, **params):
    """
    1. making item to idx
    2. padding
    :word2idx_dic
    :sequences: list of lists where each element is a sequence
    :maxlen: int, maximum length
    :dtype: type to cast the resulting sequence.
    :padding: 'pre' or 'post', pad either before or after each sequence.
    :truncating: 'pre' or 'post', remove values from sequences larger than
        maxlen either in the beginning or in the end of the sequence
    :value: float, value to pad the sequences to the desired value.
    """
    seq_idx = [[word2idx_dic.get(a, word2idx_dic['__ETC__']) for a in i]
               for i in sequences]
    params['value'] = word2idx_dic['__PAD__']
    return (pad_sequences(seq_idx, **params))


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


## 추론
def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' %
                             truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# 추론 함수 정의
def make_pred_sents(x_sents, y_pred):
    res_sent = []
    for i, j in zip(x_sents, y_pred):
        if j == 1:
            res_sent.append(i)
            res_sent.append(' ')
        else:
            res_sent.append(i)
    subs = re.sub(re.compile(r'\s+'), ' ', ''.join(res_sent).replace('^', ' '))
    subs = subs.replace('«', '')
    subs = subs.replace('»', '')
    return subs


def inference(model, input_txt, w2idx, max_seq_len, device):
    model.eval()
    input_txt_processed = pre_processing([input_txt])  # ['«아버지가^방에^들어^가^신다»']
    input_txt_processed_no_space = [tmp.replace('^', '') for tmp in input_txt_processed]  # ['«아버지가방에들어가신다»']
    # input_txt_processed_no_space = [tmp for tmp in input_txt_processed]  # ['«아버지가방에들어가신다»']

    input_txt_processed_indice = encoding_and_padding(word2idx_dic=w2idx,
                                  sequences=input_txt_processed_no_space,
                                  maxlen=max_seq_len,
                                  padding='post',
                                  truncating='post')  # [[   3,   26,  188,   12,   11,  118,    7,   29,   20,   11,   70, 2,   4, 1993, 1993,

    # print(f'{input_txt_processed}\n=>{input_txt_processed_no_space}\n=>{input_txt_processed_indice}')


    model.eval()
    input_ids = torch.tensor(input_txt_processed_indice).to(device)
        
    with torch.no_grad():
        outputs = model(input_ids)
        # Apply softmax to outputs to get probabilities
        probabilities = F.softmax(outputs, dim=-1)  # (batch_size, seq_length, num_classes)
        # Convert outputs to predicted labels by taking the argmax
        predicted_labels = torch.argmax(probabilities, dim=2)  # (batch_size, seq_length)
        predictions = predicted_labels.cpu().numpy().squeeze()

        pad_index_from = np.where(input_txt_processed_indice[0] == 1993)[0][0]
        predictions = predictions[:pad_index_from].tolist()

        result_spacing = make_pred_sents(list(input_txt_processed_no_space[0]), predictions)
        print(f'{input_txt}\n=>{input_txt_processed_no_space}\n=>{result_spacing}\n=>{predictions}')        

    return result_spacing, predictions

