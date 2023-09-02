from captioning_solver import *
from rnn import *
from load_data import *

file_name = "goblet_of_fire.txt"
word_to_idx = make_dict(file_name)

model = CaptioningRNN(word_to_idx, input_dim=512, wordvec_dim=128,\
                       hidden_dim=128, cell_type="rnn", seed=42)

# for key in model.params.keys():
#     print(key, model.params[key])

solver = CaptioningSolver(model, file_name, word_to_idx = word_to_idx,\
                         lines_count = 1, max_train = 1, batch_size =1, num_epochs = 5)
solver.train()

out = model.sample(1, 30)

reverse_dict = {}

for keys, value in word_to_idx.items():
    reverse_dict[value] = keys

for i in range(out.shape[0]):
    words = [reverse_dict[val] for val in out[i]]
    pprint(words)
    print("******")