import os
import sys
import time
from paths_hg import *
from data_loader import *

class BaseConfig:
    """
    Base configuration class to hold common parameters.
    """
    def __init__(self):
        self.model_name = "dual_dynamic_graph_context_magnn"
        self.graph_embedding_dim = 128
        self.hidden_dim = 128
        self.n_class = 1
        self.report_step_num = 10
        self.dropout_rate = 0.5
        self.min_learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 10
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2

        # Task-specific parameters
        self.text_max_length = 30
        self.pad_idx = 0 GNN
        self.use_text_features = True
        self.k_hop = 1
        self.gpu_id = "0"
        self.add_adj_size = 1
        self.magnn_num_metapaths = 3
        self.magnn_num_heads_intra = 8
        self.magnn_num_heads_inter = 8
        self.magnn_hidden_dim = 128
        self.magnn_attention_dropout = 0.5

        self.main_context_node_type_for_fusion = 'post'

class DualGraphContextMAGNNConfig(BaseConfig):
    """
    Configuration for a fusual-templ graph (propagation graph + context graph with MAGNN) model.

    """
    def __init__(self):
        super().__init__()
        # Specific overrides or additions for this configuration
        # self.batch_size = 16 # Example: can be set here or passed via args
        # self.epoch_num = 50
        # self.learning_rate = 1e-4


class DualGraphContextMAGNNConfigKO(BaseConfig): # Renamed from dual_dynamic_graph_Config_zh
    """
    Configuration for a dual graph (propagation graph + context graph with MAGNN) model,
    which can include parts specialized for Korean datasets. Currently same as BaseConfig.
    """
    def __init__(self):
        super().__init__()
        # Specific overrides or additions for Korean dataset configuration
        # self.batch_size = 16
        # self.epoch_num = 100
        # self.learning_rate = 5e-5

# --- BERT related settings can be uncommented and updated if needed ---
# class DualGraphContextMAGNNConfigBERT(BaseConfig):
#     def __init__(self):
#         super().__init__()
#         self.model_name = "dual_graph_context_magnn_bert" # Example: more specific name
#         self.use_bert_for_root = True # Whether to use BERT for root nodes
#         self.concat_root_bert_to_pooled = True # Whether to concatenate root BERT features to pooled features
#         self.freeze_bert = True # Whether to freeze BERT weights
#         self.bert_model_name_or_path = "klue/bert-base" # Example: Korean BERT model
#         # ... (Other BERT related settings like batch size, learning rate, etc.)
