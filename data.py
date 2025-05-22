import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import os
import torch
import re
import string
import torch.nn.functional as F
from paths_ko import *
import csv
from config import *
import random


def split_data_stratified(data_size, labels_tensor, train_ratio, val_ratio, test_ratio, shuffle_data=True,
                          random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    indices = np.arange(data_size)
    # Assuming binary labels 0 and 1. Adjust if labels are different.
    # Original code used 0 for positive and 1 for negative based on print statements.
    positive_indices = indices[labels_tensor.cpu().numpy().squeeze() == 0]
    negative_indices = indices[labels_tensor.cpu().numpy().squeeze() == 1]

    print(
        f"Data distribution: Positive (label 0): {len(positive_indices)}, Negative (label 1): {len(negative_indices)}")

    if shuffle_data:
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

    def _split_class_indices(class_indices, r_train, r_val, r_test):
        n = len(class_indices)
        train_end = int(r_train * n)
        val_end = train_end + int(r_val * n)
        # Ensure test gets the remainder to sum to n correctly
        return (class_indices[:train_end],
                class_indices[train_end:val_end],
                class_indices[val_end:])

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: Ratios sum to {total_ratio}, normalizing them.")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    train_pos, val_pos, test_pos = _split_class_indices(positive_indices, train_ratio, val_ratio, test_ratio)
    train_neg, val_neg, test_neg = _split_class_indices(negative_indices, train_ratio, val_ratio, test_ratio)

    train_indices = np.concatenate((train_pos, train_neg))
    val_indices = np.concatenate((val_pos, val_neg))
    test_indices = np.concatenate((test_pos, test_neg))

    if shuffle_data:  # Shuffle combined indices again
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

    print(f"Train split: Positive: {len(train_pos)}, Negative: {len(train_neg)}, Total: {len(train_indices)}")
    print(f"Validation split: Positive: {len(val_pos)}, Negative: {len(val_neg)}, Total: {len(val_indices)}")
    print(f"Test split: Positive: {len(test_pos)}, Negative: {len(test_neg)}, Total: {len(test_indices)}")

    return train_indices, val_indices, test_indices


def split_data_early_stratified(data_size, labels_tensor, train_ratio, shuffle_data=True, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    indices = np.arange(data_size)
    positive_indices = indices[labels_tensor.cpu().numpy().squeeze() == 0]
    negative_indices = indices[labels_tensor.cpu().numpy().squeeze() == 1]

    if shuffle_data:
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

    def _split_class_train_val(class_indices, r_train):
        n = len(class_indices)
        train_end = int(r_train * n)
        return class_indices[:train_end], class_indices[train_end:]

    train_pos, val_pos = _split_class_train_val(positive_indices, train_ratio)
    train_neg, val_neg = _split_class_train_val(negative_indices, train_ratio)

    train_indices = np.concatenate((train_pos, train_neg))
    val_indices = np.concatenate((val_pos, val_neg))

    if shuffle_data:
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    return train_indices, val_indices


def split_data_5fold_stratified(data_size, labels_tensor, shuffle_data=True, random_seed=None):

    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        raise ImportError("scikit-learn is required for 5-fold stratified splitting. Please install it.")

    indices = np.arange(data_size)
    labels_numpy = labels_tensor.cpu().numpy().ravel()  # StratifiedKFold expects 1D array for y

    skf = StratifiedKFold(n_splits=5, shuffle=shuffle_data,
                          random_state=random_seed if shuffle_data else None)

    print(
        f"Original data distribution for 5-fold: Positive (label 0): {np.sum(labels_numpy == 0)}, Negative (label 1): {np.sum(labels_numpy == 1)}")

    fold_splits = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels_numpy)):
        fold_splits.append({'train_val': train_val_idx, 'test': test_idx})
        print(
            f"Fold {fold_idx + 1}: Test Positive={np.sum(labels_numpy[test_idx] == 0)}, Negative={np.sum(labels_numpy[test_idx] == 1)}")
    return fold_splits


def sparse_to_tuple(sparse_mx):

    def _to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        return [_to_tuple(m) for m in sparse_mx]
    else:
        return _to_tuple(sparse_mx)


def normalize_adjacency_matrix(adj, add_self_loops=True):  # Renamed from normalize_adj

    if add_self_loops:
        adj_processed = adj + sp.eye(adj.shape[0])
    else:
        adj_processed = adj.copy()

    adj_processed = sp.coo_matrix(adj_processed)
    row_sum = np.array(adj_processed.sum(1))

    with np.errstate(divide='ignore'):  # Suppress RuntimeWarning for division by zero
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = adj_processed.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_adj.tocoo()


def row_normalize_features(mx):  # Renamed from normalize
    row_sum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(row_sum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    normalized_mx = r_mat_inv.dot(mx)
    return normalized_mx


def scipy_sparse_to_torch_sparse(sparse_mx):  # Renamed from sparse_mx_to_torch_sparse_tensor
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def scipy_sparse_to_torch_edges(sparse_mx):  # Renamed from sparse_mx_to_torch
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if np.any(np.isnan(sparse_mx.data)):
        print("Warning: NaN values found in sparse matrix data. Replacing with 0.")
        sparse_mx.data = np.nan_to_num(sparse_mx.data)  # Replace NaNs with 0

    edge_index = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    edge_attr = torch.from_numpy(sparse_mx.data.astype(np.float32))  # Ensure float32 for attributes
    return edge_index, edge_attr, torch.Size(sparse_mx.shape)


def check_file_exists(filepath):  # Renamed from check_exist
    return os.path.isfile(filepath)


def load_word2vec_embeddings(file_path):  # Renamed from load_w2v_emb
    print(f'Loading Word2Vec embeddings from: {file_path}')
    if not check_file_exists(file_path):
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    with open(file_path, 'rb') as f:
        embeddings = np.load(f)
    return embeddings  # Should be a numpy array


def pad_text_sequence(text_id_list, max_length, pad_token_id):  # Renamed from text_length_pad

    if len(text_id_list) > max_length:
        padded_list = text_id_list[:max_length]
    else:
        padded_list = text_id_list + [pad_token_id] * (max_length - len(text_id_list))
    assert len(padded_list) == max_length
    return padded_list
