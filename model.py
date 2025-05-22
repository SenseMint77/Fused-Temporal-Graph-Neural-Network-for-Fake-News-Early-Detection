from __future__ import absolute_import, unicode_literals, division, print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
# from torch.nn.parameter import Parameter # Not directly used
# from torch.nn.modules.module import Module # Base class for nn.Module
import torch.nn.functional as F
# from torch_sparse import spmm # Potentially used in custom GCN layers if not using PyG's
# Assuming layers.py contains the refactored layer classes
from layers import (PropagationGCNLayer, ContextGraphMAGNNLayer,
                    TemporalFusionLayer,  # KnowledgeGCNLayer replaced with ContextGraphMAGNNLayer
                    GlobalMeanPool, GRUTextEncoder, BiLSTMTextEncoder, ScaledDotProductAttention)
# from tqdm import tqdm # tqdm is for progress bars, usually in training scripts, not model definitions
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModel


# --- Base Model Structure ---
class DualGraphRumorModelBase(nn.Module):
    """
    Base class for Dual Dynamic Graph models for rumor detection.
    This class provides a common structure for feature extraction, graph neural networks (GCN for propagation, HIN-GNN for context),
    cross-graph fusion, and classification.
    Specific variations can inherit and override components.
    """

    def __init__(self, config, text_vocab_size, graph_node_vocab_size,
                 device):  # For HIN, graph_node_vocab_size might need to consider vocabulary size per type
        super(DualGraphRumorModelBase, self).__init__()
        self.config = config
        self.device = device  # Usually model.to(device) is called outside

        # === Feature Extractors ===
        # 1. Text Embedding for non-BERT node content (e.g., user descriptions, non-root posts)
        self.text_embedding_dim = config.graph_embedding_dim  # Can be different from GCN hidden_dim
        if text_vocab_size > 0:
            self.text_node_embedding = nn.Embedding(text_vocab_size, self.text_embedding_dim,
                                                    padding_idx=config.pad_idx)
        else:
            self.text_node_embedding = None
            print("Warning: text_vocab_size is 0, text_node_embedding will not be created.")

        # 2. Node ID Embedding (fallback or primary for graph structure if no rich features)
        #    For context graph (HIN), embeddings for various node types might also need to be considered.
        #    E.g., receive graph_node_vocab_size as a dictionary of vocabulary sizes by type, or use a common ID space
        if graph_node_vocab_size > 0:  # Assuming single homogeneous graph ID embedding. Needs modification for HIN.
            self.graph_node_id_embedding = nn.Embedding(graph_node_vocab_size, config.graph_embedding_dim)
        else:
            self.graph_node_id_embedding = None
            print("Warning: graph_node_vocab_size is 0, graph_node_id_embedding will not be created.")

        # 3. RNN for sequential text data (optional)
        self.text_rnn_encoder = None
        self.rnn_output_dim = 0
        if getattr(config, 'use_text_rnn', False) and self.text_node_embedding:
            rnn_type = getattr(config, 'text_rnn_type', 'gru').lower()
            bidirectional_rnn = getattr(config, 'text_rnn_bidirectional', True)
            if rnn_type == 'gru':
                self.text_rnn_encoder = GRUTextEncoder(self.text_embedding_dim, config.hidden_dim,
                                                       bidirectional=bidirectional_rnn)
            elif rnn_type == 'lstm':
                self.text_rnn_encoder = BiLSTMTextEncoder(self.text_embedding_dim,
                                                          config.hidden_dim)  # BiLSTMTextEncoder is already bidirectional
            if self.text_rnn_encoder:
                self.rnn_output_dim = config.hidden_dim * (
                    2 if bidirectional_rnn and rnn_type == 'gru' else (2 if rnn_type == 'lstm' else 1))

        # 4. BERT model for text features (optional, typically for root post content)
        self.bert_model = None
        self.bert_projector = None  # To project BERT embeddings to hidden_dim
        self.bert_output_dim = config.hidden_dim  # Target dimension after projection
        if getattr(config, 'use_bert_for_root', False) and hasattr(config, 'bert_model_name_or_path'):
            try:
                bert_config_huggingface = BertConfig.from_pretrained(config.bert_model_name_or_path,
                                                                     output_hidden_states=True)
                self.bert_model = BertModel.from_pretrained(config.bert_model_name_or_path,
                                                            config=bert_config_huggingface)
                if getattr(config, 'freeze_bert', True):  # Default to freezing BERT
                    for param in self.bert_model.parameters():
                        param.requires_grad = False
                self.bert_projector = nn.Linear(self.bert_model.config.hidden_size, self.bert_output_dim)
            except Exception as e:
                print(f"Error loading BERT model '{config.bert_model_name_or_path}': {e}")
                self.bert_model = None

        self.propagation_gcn_feature_dim = config.hidden_dim  # For propagation graph GCN
        self.context_graph_feature_dim = config.hidden_dim  # For context graph HIN-GNN (can be same or different)

        # === Graph Neural Networks ===
        self.num_gnn_layers = getattr(config, 'num_gnn_layers', 3)  # Number of GNN layers
        self.propagation_gcn_stack = nn.ModuleList()

        # MAGNN-based GNN stack for context graph (HIN)
        self.context_magnn_stack = nn.ModuleList()

        # === Feature Fusion ===
        self.use_cross_graph_fusion = getattr(config, 'use_cross_graph_fusion', True)
        if self.use_cross_graph_fusion:
            self.fusion_stack = nn.ModuleList()

        # Batch Normalization (optional)
        self.use_batch_norm = getattr(config, 'use_batch_norm', True)
        if self.use_batch_norm:
            self.bn_propagation_stack = nn.ModuleList()
            self.bn_context_stack = nn.ModuleList()  # BN for context graph

        # MAGNN layer settings (should be from config)
        # E.g., num_metapaths, num_heads_inter_metapath, etc.
        magnn_num_metapaths = getattr(config, 'magnn_num_metapaths', 3)  # Example: Number of metapaths
        magnn_num_heads = getattr(config, 'magnn_num_heads', 8)  # Example: Number of attention heads

        for _ in range(self.num_gnn_layers):
            self.propagation_gcn_stack.append(
                PropagationGCNLayer(self.propagation_gcn_feature_dim, self.propagation_gcn_feature_dim,
                                    self.propagation_gcn_feature_dim,
                                    config.dropout_rate, device)
            )
            # MAGNN layer for context graph (HIN)
            # ContextGraphMAGNNLayer should be defined in layers.py and encapsulate MAGNN's logic
            # (Metapath instance encoding, intra-metapath attention, inter-metapath attention)
            self.context_magnn_stack.append(
                ContextGraphMAGNNLayer(
                    # Since MAGNN can take node features of various types as input,
                    # in_channels can be a dictionary type, or a dimension after all node features are converted to a common dimension.
                    # Here, assuming use of common dimension for simplification.
                    in_channels=self.context_graph_feature_dim,
                    out_channels=self.context_graph_feature_dim,
                    num_metapaths=magnn_num_metapaths,  # Needs to be defined in config
                    num_heads=magnn_num_heads,  # Needs to be defined in config
                    dropout_rate=config.dropout_rate
                )
            )

            if self.use_cross_graph_fusion:
                # Input dimension of fusion layer considers propagation_gcn_feature_dim and context_graph_feature_dim
                # Here, assuming they are the same
                self.fusion_stack.append(TemporalFusionLayer(self.propagation_gcn_feature_dim))
            if self.use_batch_norm:
                self.bn_propagation_stack.append(nn.BatchNorm1d(self.propagation_gcn_feature_dim))
                self.bn_context_stack.append(nn.BatchNorm1d(self.context_graph_feature_dim))

        # === Pooling & Classifier ===
        self.global_pool = GlobalMeanPool()

        classifier_input_dim = self.propagation_gcn_feature_dim
        if getattr(config, 'concat_root_bert_to_pooled', False) and self.bert_model:
            classifier_input_dim += self.bert_output_dim

        self.classifier_fc1 = nn.Linear(classifier_input_dim, getattr(config, 'classifier_hidden_dim', 100))
        self.classifier_fc2 = nn.Linear(getattr(config, 'classifier_hidden_dim', 100), config.n_class)

    def _prepare_initial_node_features(self, data, graph_type):
        """
        Abstract method to prepare initial node features for GCN/HIN-GNN layers.
        Subclasses must implement this based on their specific feature strategy.
        This method should return a tensor of shape (num_total_nodes_in_batch_for_graph_type, feature_dim_of_the_gnn).
        For context graph (HIN), the data object can be of HeteroData type and may include features by node type.

        Args:
            data (torch_geometric.data.Batch or torch_geometric.data.HeteroData): Batch of graph data.
            graph_type (str): 'propagation' or 'context'.

        Returns:
            torch.Tensor or Dict[str, torch.Tensor]: Initial node features. Can be a dictionary of features by type for HIN.
        """
        raise NotImplementedError("Subclasses must implement _prepare_initial_node_features.")

    def _get_bert_features_for_roots(self, data):
        """
        Extracts BERT features for root nodes if BERT model is available.
        Assumes `data` contains `root_input_ids` and `root_attention_mask`.
        """
        if not (self.bert_model and hasattr(data, 'root_input_ids') and hasattr(data, 'root_attention_mask')):
            return None

        bert_outputs = self.bert_model(
            input_ids=data.root_input_ids.to(self.device),
            attention_mask=data.root_attention_mask.to(self.device)
        )
        pooled_bert_output = bert_outputs.pooler_output
        projected_bert_features = self.bert_projector(pooled_bert_output)
        return F.relu(projected_bert_features)

    def forward(self, data):
        """
        Main forward pass for the model.
        `data` is a PyTorch Geometric Batch or HeteroData object.
        - For propagation graph: data.x_prop (initial features), data.edge_index_prop, data.batch_prop, data.root_indices_prop.
        - For context graph (HIN):
            - data['node_type'].x (features by node type),
            - data[('src_type', 'edge_type', 'dst_type')].edge_index (connectivity by edge type),
            - data.metapath_dict (metapath information for MAGNN).
            - data.batch_context (or batch vector by type).
        - For fusion: data.common_indices_prop, data.common_indices_context, data.initial_features_common_nodes.
        - For root BERT features: data.root_input_ids, data.root_attention_mask.
        """
        data = data.to(self.device)

        # 1. Get initial node features for both graphs
        x_prop = self._prepare_initial_node_features(data, 'propagation')
        x_context_initial = self._prepare_initial_node_features(data, 'context')

        initial_prop_features_clone = x_prop.clone() if self.use_cross_graph_fusion else None

        # 2. Multi-layer GNN processing
        current_x_prop = x_prop
        current_x_context = x_context_initial

        for i in range(self.num_gnn_layers):
            # Propagation GCN
            x_prop_out = self.propagation_gcn_stack[i](
                current_x_prop, data.edge_index_prop, getattr(data, 'edge_weight_prop', None),
                data.root_indices_prop,
                data.batch_prop
            )
            if self.use_batch_norm:
                x_prop_out = self.bn_propagation_stack[i](x_prop_out)
            x_prop_activated = F.leaky_relu(x_prop_out)
            x_prop_next_input = F.dropout(x_prop_activated, p=self.config.dropout_rate, training=self.training)

            # Context Graph MAGNN Layer
            # Need to pass information required for MAGNN from the data object
            x_context_out = self.context_magnn_stack[i](
                current_x_context,
                data  # Pass the entire data object so the layer can extract necessary info
            )

            if self.use_batch_norm:
                if isinstance(x_context_out, dict):
                    temp_context_out = {ntype: self.bn_context_stack[i](feat) for ntype, feat in x_context_out.items()}
                    x_context_out = temp_context_out
                else:
                    x_context_out = self.bn_context_stack[i](x_context_out)

            if isinstance(x_context_out, dict):
                x_context_activated = {ntype: F.leaky_relu(feat) for ntype, feat in x_context_out.items()}
            else:
                x_context_activated = F.leaky_relu(x_context_out)

            if isinstance(x_context_activated, dict):
                x_context_next_input = {ntype: F.dropout(feat, p=self.config.dropout_rate, training=self.training) for
                                        ntype, feat in x_context_activated.items()}
            else:
                x_context_next_input = F.dropout(x_context_activated, p=self.config.dropout_rate,
                                                 training=self.training)

            # Cross-graph fusion
            if self.use_cross_graph_fusion and hasattr(data, 'common_indices_prop') and hasattr(data,
                                                                                                'common_indices_context'):
                if hasattr(data, 'initial_features_common_nodes'):
                    h_original_common = data.initial_features_common_nodes
                else:
                    h_original_common = initial_prop_features_clone[data.common_indices_prop]

                x_context_for_fusion = x_context_activated
                if isinstance(x_context_activated, dict):
                    main_context_node_type = getattr(self.config, 'main_context_node_type_for_fusion', 'post')
                    if main_context_node_type in x_context_activated:
                        x_context_for_fusion = x_context_activated[main_context_node_type]
                    else:
                        print(
                            f"Warning: Features for main context node type '{main_context_node_type}' for fusion not found.")
                        x_context_for_fusion = torch.zeros_like(x_prop_activated)

                prop_fused, context_fused = self.fusion_stack[i](
                    x_prop_activated,
                    x_context_for_fusion,
                    h_original_common,
                    data.common_indices_prop,
                    data.common_indices_context
                )
                current_x_prop = F.dropout(prop_fused, p=self.config.dropout_rate, training=self.training)
                current_x_context = F.dropout(context_fused, p=self.config.dropout_rate, training=self.training)
            else:
                current_x_prop = x_prop_next_input
                current_x_context = x_context_next_input

        # 3. Global Pooling
        pooled_graph_features = self.global_pool(current_x_prop, data.batch_prop)

        # 4. Optional: Concatenate root BERT features
        final_classifier_input = pooled_graph_features
        if getattr(self.config, 'concat_root_bert_to_pooled', False) and self.bert_model:
            root_bert_features = self._get_bert_features_for_roots(data)
            if root_bert_features is not None:
                final_classifier_input = torch.cat([pooled_graph_features, root_bert_features], dim=1)

        # 5. Classifier
        hidden_logits = F.leaky_relu(self.classifier_fc1(final_classifier_input))
        hidden_logits_dropped = F.dropout(hidden_logits, p=self.config.dropout_rate, training=self.training)
        output_logits = self.classifier_fc2(hidden_logits_dropped)

        if self.config.n_class == 1:
            return torch.sigmoid(output_logits)
        else:
            return output_logits


# --- Example Specific Model Implementation ---
class CompletedGRUWithoutBERTModel(DualGraphRumorModelBase):
    """
    Specific model variant: uses GRU for initial text node features,
    dual GNNs (assuming GCN + HIN-GNN), feature fusion,
    but NO final BERT feature concatenation for classification.
    This corresponds to the logic of "dual_DynamicGCN_nBatch_completed_gru_wobert".
    """

    def __init__(self, config, text_vocab_size, graph_node_vocab_size, device, mid_to_token_ids_map):
        # Ensure 'use_text_rnn' and 'text_rnn_type' are set in config for this model type
        config.use_text_rnn = True
        config.text_rnn_type = 'gru'
        config.text_rnn_bidirectional = True

        # This model variant does NOT use BERT for root features in the final classifier.
        config.use_bert_for_root = False
        config.concat_root_bert_to_pooled = False

        super(CompletedGRUWithoutBERTModel, self).__init__(config, text_vocab_size, graph_node_vocab_size, device)

        # Store the mapping from original node ID (MID) to its token ID sequence
        # This map should be precomputed by the data processor.
        self.mid_to_token_ids_map = mid_to_token_ids_map

        # Project RNN output to propagation_gcn_feature_dim if they are different
        if self.text_rnn_encoder and self.rnn_output_dim != self.propagation_gcn_feature_dim:
            self.rnn_to_gcn_projector = nn.Linear(self.rnn_output_dim, self.propagation_gcn_feature_dim)
        else:
            self.rnn_to_gcn_projector = None

        # Feature projector for context graph (HIN) nodes (if using ID embedding)
        if self.graph_node_id_embedding and self.config.graph_embedding_dim != self.context_graph_feature_dim:
            self.context_id_embed_projector = nn.Linear(self.config.graph_embedding_dim, self.context_graph_feature_dim)
        else:
            self.context_id_embed_projector = None

    def _prepare_initial_node_features(self, data, graph_type):
        """
        Uses GRU-processed text embeddings for nodes with text (mainly for propagation graph),
        and ID embeddings (mainly for context graph or as fallback) to prepare initial node features.
        In actual HIN, feature initialization strategy for different node types might be needed.
        """
        # === Very Important: The loop below is for conceptual explanation and must be optimized for batch processing in actual implementation ===
        # It is recommended to prepare features in advance using PyG's `data.node_store` (for HeteroData) or `transform`.

        if graph_type == 'propagation':
            # data.original_p_node_ids should be a flat tensor of original MIDs for all nodes in the prop batch
            # These IDs are used as keys for self.mid_to_token_ids_map
            original_node_ids = data.original_p_node_ids  # Get from PyG Data object (needs to be defined in data_process.py)
            target_feature_dim = self.propagation_gcn_feature_dim

            # Propagation node features (GRU + text embedding)
            batch_node_features = []
            for node_id_tensor in original_node_ids:
                node_id = node_id_tensor.item()
                node_feat = None
                if node_id in self.mid_to_token_ids_map and self.text_node_embedding and self.text_rnn_encoder:
                    token_ids = torch.tensor(self.mid_to_token_ids_map[node_id], dtype=torch.long).unsqueeze(0).to(
                        self.device)
                    embedded_text = self.text_node_embedding(token_ids)
                    node_feat_rnn = torch.mean(embedded_text, dim=1).squeeze(0)  # Temporary mean pooling
                    if self.rnn_to_gcn_projector:
                        node_feat_rnn = self.rnn_to_gcn_projector(node_feat_rnn.unsqueeze(0)).squeeze(0)
                    node_feat = node_feat_rnn
                elif self.graph_node_id_embedding:  # Fallback to ID embedding for nodes without text
                    if node_id < self.graph_node_id_embedding.num_embeddings:
                        node_feat = self.graph_node_id_embedding(
                            torch.tensor(node_id, dtype=torch.long).to(self.device))
                        if node_feat.shape[
                            -1] != target_feature_dim:  # Project if dims don't match (temp projector needed)
                            node_feat = torch.zeros(target_feature_dim, device=self.device)  # Temporary
                    else:
                        node_feat = torch.zeros(target_feature_dim, device=self.device)
                else:
                    node_feat = torch.zeros(target_feature_dim, device=self.device)
                batch_node_features.append(node_feat)

            return torch.stack(batch_node_features) if batch_node_features else torch.empty((0, target_feature_dim),
                                                                                            device=self.device)

        elif graph_type == 'context':
            # For context graph (HIN), if using HeteroData, features by type might already be in data['node_type'].x,
            # or generate them by type here. MAGNN can take a dictionary of features by type as input.
            # Below is a simplified example assuming all context nodes use ID embedding.
            # In reality, type-specific feature initialization is needed.
            if hasattr(data, 'x_context_node_types') and hasattr(data,
                                                                 'x_context_original_ids') and self.graph_node_id_embedding:
                # data.x_context_node_types: type of each node (['user', 'post', 'topic', ...])
                # data.x_context_original_ids: original ID of each node (or globally mapped ID)
                # This example processes all types with the same ID embedding. In reality, type-specific embedding/projection is needed.
                node_features_dict = {}
                # Assuming data.x_context_node_types_tensor is available
                unique_node_types = torch.unique(data.x_context_node_types_tensor)

                for node_type_val_tensor in unique_node_types:
                    node_type_val = node_type_val_tensor.item()  # Get Python int/str for dict key
                    node_type_str = str(node_type_val)  # Ensure string key

                    type_mask = (data.x_context_node_types_tensor == node_type_val_tensor)
                    type_original_ids = data.x_context_original_ids[type_mask]  # Node IDs of this type

                    # This part greatly depends on the actual HIN data structure and feature preparation method
                    # Temporarily, assume all types are processed with graph_node_id_embedding
                    if type_original_ids.numel() > 0:
                        # This assumes type_original_ids are already mapped to valid embedding indices
                        embedded_features = self.graph_node_id_embedding(type_original_ids)
                        if self.context_id_embed_projector:
                            node_features_dict[node_type_str] = self.context_id_embed_projector(embedded_features)
                        else:  # Ensure dimension matches context_graph_feature_dim
                            if embedded_features.shape[-1] != self.context_graph_feature_dim:
                                print(
                                    f"Warning: Dimension mismatch for context node type '{node_type_str}'. Expected {self.context_graph_feature_dim}, got {embedded_features.shape[-1]}. Using zeros.")
                                node_features_dict[node_type_str] = torch.zeros(
                                    (type_original_ids.size(0), self.context_graph_feature_dim), device=self.device)
                            else:
                                node_features_dict[node_type_str] = embedded_features
                    else:
                        node_features_dict[node_type_str] = torch.empty((0, self.context_graph_feature_dim),
                                                                        device=self.device)
                return node_features_dict  # MAGNN should be able to process feature dictionary by type
            else:
                print("Warning: Cannot prepare initial context graph features. (HeteroData structure check needed)")
                num_context_nodes = data.num_nodes_context_total if hasattr(data, 'num_nodes_context_total') else 0
                return torch.zeros((num_context_nodes, self.context_graph_feature_dim),
                                   device=self.device)  # Example of returning a single tensor
        else:
            raise ValueError("Invalid graph_type")