import numpy as np
import random
import time
from pprint import pprint as pprint

def make_dict(name):
    file = open(name, 'r')
    lines = file.readlines()
    dict = {}
    idx = 0

    for i, line in enumerate(lines):
        words = line.split()
        for word in words:
            try:
                _ = dict[word]
            except:
                dict[word] = idx
                idx += 1

    dict['<START>'] = idx 
    dict['<NULL>'] = idx + 1
    dict['<END>'] = idx + 2

    return  dict

def make_dict_letter(name):
    file = open(name, 'r')
    lines = file.readlines()
    dict = {}
    idx = 0

    for i, line in enumerate(lines):
        for letter in line:
            try:
                _ = dict[letter]
            except:
                dict[letter] = idx
                idx += 1

    dict['<START>'] = idx
    dict['<NULL>'] = idx + 1
    dict['<END>'] = idx + 2

    return dict

def load_data(dict, name, lines_count = 5, max_train = 50, hardcode = None):

    np.random.seed(int(time.time()))

    start_token = dict["<START>"]
    null_token = dict["<NULL>"]
    end_token = dict["<END>"]

    file = open(name, 'r')
    lines = file.readlines()
    num_lines = len(lines)

    start_indices = np.random.randint(num_lines - lines_count - 1, size = max_train)

    if(hardcode is not None):
        start_indices = np.ones((max_train), dtype=np.int32) * hardcode

    max_words = 0
    for i in range(max_train):
        para = lines[start_indices[i]:start_indices[i]+lines_count]
        curr = 0
        for j in range(lines_count):
            curr += len(para[j].split())
        max_words = max(max_words, curr)
    
    batch = np.zeros((max_train, max_words + 2), dtype=np.int32)

    for i in range(max_train):

        para = lines[start_indices[i]:start_indices[i]+lines_count]
        curr_set = [start_token]
        num_words = 0

        for j in range(lines_count):
            words = para[j].split()
            curr_set += [dict[word] for word in words]
            num_words += len(words)

        curr_set.append(end_token)

        for j in range(max_words - num_words):
            curr_set.append(null_token)
        
        batch[i,:] = np.array(curr_set, dtype=np.int32)

    return batch

def load_data_letter(dict, name, lines_count = 5, max_train = 50, hardcode = None):

    np.random.seed(int(time.time()))

    start_token = dict["<START>"]
    null_token = dict["<NULL>"]
    end_token = dict["<END>"]

    file = open(name, 'r')
    lines = file.readlines()
    num_lines = len(lines)

    start_indices = np.random.randint(num_lines - lines_count - 1, size = max_train)

    if(hardcode is not None):
        start_indices = np.ones((max_train), dtype=np.int32) * hardcode
    
    max_letters = 0
    for i in range(max_train):
        para = lines[start_indices[i]:start_indices[i]+lines_count]
        curr = 0
        for j in range(lines_count):
            curr += len(para[j])
        max_letters = max(max_letters, curr)
    
    batch = np.zeros((max_train, max_letters + 2), dtype=np.int32)

    for i in range(max_train):

        para = lines[start_indices[i]:start_indices[i]+lines_count]
        curr_set = [start_token]
        num_letters = 0

        for j in range(lines_count):
            curr_set += [dict[letter] for letter in para[j]]
            num_letters += len(para[j])

        curr_set.append(end_token)

        for j in range(max_letters - num_letters):
            curr_set.append(null_token)
        
        batch[i,:] = np.array(curr_set, dtype=np.int32)

    return batch


def load_data_letter_2_digit_ops(dict, name, lines_count = 1, max_train = 50, hardcode = None):

    np.random.seed(int(time.time()))

    file = open(name, 'r')
    lines = file.readlines()
    num_lines = len(lines)

    start_indices = np.random.randint(num_lines - lines_count - 1, size = max_train)

    if(hardcode is not None):
        start_indices = np.ones((max_train), dtype=np.int32) * hardcode

    batch = np.zeros((max_train, 15), dtype=np.int32)

    for i in range(max_train):

        para = lines[start_indices[i]]
        curr_set = []
        curr_set += [dict[letter] for letter in para]
        
        batch[i,:] = np.array(curr_set, dtype=np.int32)
        # batch[i,0:9] = dict["<NULL>"]

    return batch
