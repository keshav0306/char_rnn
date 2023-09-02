import torch
import torch.nn as nn
from load_data import *
import numpy as np
from rnn_layers_torch import *

class RNN(nn.Module):

    def __init__(self, word_to_idx, wordvec_dim, hidden_dim, cell_type, seed):
        super(RNN, self).__init__()

        vocab_size = len(word_to_idx)
        self.start_token = word_to_idx["<START>"]
        self.null_token = word_to_idx["<NULL>"]
        self.end_token = word_to_idx["<END>"]
        self.cell_type = cell_type
        self.params = {}

        if(seed is not None):
            np.random.seed(seed)

        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        self.params["h_init"] = np.random.randn(hidden_dim)

        for key in self.params.keys():
            self.params[key] = self.params[key].astype(np.float32)
            self.params[key] = torch.from_numpy(self.params[key])
            self.params[key].requires_grad = True


        # self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim, requires_grad=True) / 100

        # dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        # self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim, requires_grad=True)
        # self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim,requires_grad=True)
        # self.params["b"] = torch.zeros(dim_mul * hidden_dim,requires_grad=True)
        # self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size,requires_grad=True)
        # self.params["b_vocab"] = torch.zeros(vocab_size, requires_grad=True)
        # self.params["h_init"] = torch.randn(hidden_dim, requires_grad=True)

    def forward(self, captions):

        captions_out = captions[:,1:]
        captions_in = captions[:,:-1]
        mask = captions_out != self.null_token
        N = captions.shape[0]
        h0 = torch.tile(self.params["h_init"], (N, 1))
        h = None

        inputs = word_embedding_forward(captions_in, self.params["W_embed"])
        if(self.cell_type == "rnn"):
            h = rnn_forward(inputs, h0, self.params["Wx"], self.params["Wh"], self.params["b"])
        elif(self.cell_type == "lstm"):
            h = lstm_forward(inputs, h0, self.params["Wx"], self.params["Wh"], self.params["b"])
        else:
            return None
        out = temporal_affine_forward(h, self.params["W_vocab"], self.params["b_vocab"])
        loss = temporal_softmax_loss(out, captions_out, mask)

        return loss
    
file_name = "goblet_of_fire.txt"
word_to_idx = make_dict(file_name)
reverse_dict = {}

for keys, value in word_to_idx.items():
    reverse_dict[value] = keys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

rnn = RNN(word_to_idx, 128,128, "lstm", seed=42)

epochs = 500
learning_rate = 0.01

data = load_data(word_to_idx, file_name, lines_count=2, max_train=4)

for i in range(len(data)):
    words = [reverse_dict[val] for val in data[i]]
    print(words)
    print("******")

data = torch.from_numpy(data)

for i in range(epochs):
    loss = rnn(data)
    print(loss)
    for keys in rnn.params.keys():
        rnn.params[keys].retain_grad()
    loss.backward(retain_graph=True)
    with torch.no_grad():
        for key, value in rnn.params.items():
            rnn.params[key] -= learning_rate * rnn.params[key].grad
            rnn.params[key].grad.zero_()


start_weights = rnn.params['W_embed'][rnn.start_token]
start_weights = torch.tile(start_weights, (1, 1))

prev_h = torch.tile(rnn.params["h_init"], (1, 1))
prev_c = torch.zeros((1, prev_h.shape[1]))
curr_x = start_weights
next_h, next_c = None, None
max_length = 30
captions = rnn.null_token * torch.ones((1, max_length), dtype=torch.int32)
prev_c = torch.zeros((1, prev_h.shape[1]))

for i in range(max_length):
    if(rnn.cell_type == "rnn"):
        next_h = rnn_step_forward(curr_x, prev_h, rnn.params["Wx"], rnn.params["Wh"], rnn.params["b"])
    else:
        next_h, next_c = lstm_step_forward(curr_x, prev_h, prev_c, rnn.params["Wx"], rnn.params["Wh"], rnn.params["b"])
    # print(next_h)
    out = affine_forward(next_h, rnn.params["W_vocab"], rnn.params["b_vocab"])
    indices = torch.argmax(out, dim=1)
    captions[:, i] = indices
    prev_h = next_h
    prev_c = next_c
    curr_x = rnn.params["W_embed"][indices]

captions = captions.tolist()


for i in range(len(captions)):
    words = [reverse_dict[val] for val in captions[i]]
    print(words)
    print("******")
