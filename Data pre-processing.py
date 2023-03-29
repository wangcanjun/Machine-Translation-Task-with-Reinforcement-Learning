#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import torch

def load_data(source_path, target_path, max_length):
    source_sentences = []
    target_sentences = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = re.findall(r'\w+|[^\w\s]', line, re.UNICODE)
            if len(words) > max_length:
                words = words[:max_length]
            source_sentences.append(words)
    with open(target_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = re.findall(r'\w+|[^\w\s]', line, re.UNICODE)
            if len(words) > max_length:
                words = words[:max_length]
            target_sentences.append(words)

    source_word2idx = {}
    source_idx2word = {}
    target_word2idx = {}
    target_idx2word = {}

    for sentence in source_sentences:
        for word in sentence:
            if word not in source_word2idx:
                idx = len(source_word2idx)
                source_word2idx[word] = idx
                source_idx2word[idx] = word

    for sentence in target_sentences:
        for word in sentence:
            if word not in target_word2idx:
                idx = len(target_word2idx)
                target_word2idx[word] = idx
                target_idx2word[idx] = word

    source_seqs = []
    target_seqs = []
    for source_sentence, target_sentence in zip(source_sentences, target_sentences):
        source_seq = [source_word2idx[word] for word in source_sentence]
        target_seq = [target_word2idx[word] for word in target_sentence]
        source_seqs.append(source_seq)
        target_seqs.append(target_seq)

    return source_seqs, target_seqs, source_word2idx, source_idx2word, target_word2idx, target_idx2word

def pad_sequence(seq, max_length, pad_value):
    padded_seq = seq[:max_length] + [pad_value] * max(0, max_length - len(seq))
    return padded_seq

def preprocess_data(source_path, target_path, max_length, pad_value):
    source_seqs, target_seqs, source_word2idx, source_idx2word, target_word2idx, target_idx2word = load_data(source_path, target_path, max_length)
    source_seqs = [pad_sequence(seq, max_length, pad_value) for seq in source_seqs]
    target_seqs = [pad_sequence(seq, max_length, pad_value) for seq in target_seqs]
    source_seqs = torch.LongTensor(source_seqs)
    target_seqs = torch.LongTensor(target_seqs)
    return source_seqs, target_seqs, source_word2idx, source_idx2word, target_word2idx, target_idx2word

