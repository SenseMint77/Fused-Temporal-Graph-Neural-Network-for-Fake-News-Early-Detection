from __future__ import absolute_import, unicode_literals, division, print_function

import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_sparse import spmm  # Used for sparse matrix multiplication
from torch_geometric.nn import GCNConv  # GCN layer from PyTorch Geometric
from torch_scatter import scatter_mean  # Used for global pooling


# --- Original GCN Layer (for reference or for homogeneous graph parts) ---
class OriginalGraphConvolution(Module):  # Renamed from GraphConvolution
    """
    Original custom Graph Convolution layer.
    Consider replacing with torch_geometric.nn.GCNConv for efficiency and standard features.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(OriginalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, adj_matrix_sparse):  # adj should be a sparse tensor
        support = torch.mm(node_features, self.weight)  # XW
        if adj_matrix_sparse.is_sparse:
            output = spmm(adj_matrix_sparse._indices(), adj_matrix_sparse._values(),
                          adj_matrix_sparse.size(0), adj_matrix_sparse.size(1), support)  # AXW
        else:
            print("Warning: adj_matrix_sparse is not a sparse tensor in OriginalGraphConvolution. Using dense mm.")
            output = torch.mm(adj_matrix_sparse, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"


# --- Propagation Graph Convolution Layer ---
class PropagationGCNLayer(Module):
    """
    GCN layer for propagation graphs, incorporating root node information.
    Uses PyTorch Geometric's GCNConv.
    """

    def __init__(self, in_features, hidden_features, out_features, dropout_rate, device):
        super(PropagationGCNLayer, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(hidden_features + in_features, out_features, add_self_loops=False, normalize=False)
        self.output_transform_linear = nn.Linear(out_features + hidden_features, in_features)
        self.dropout_rate = dropout_rate
        self.device = device

    def forward(self, x_node_features, edge_index, edge_weight, root_indices_in_batch, batch_vector):
        initial_features_copy = x_node_features.clone()
        h_conv1 = self.conv1(x_node_features, edge_index, edge_weight=edge_weight)
        h_conv1_activated = F.leaky_relu(h_conv1)
        root_features_to_concat_conv1 = initial_features_copy[root_indices_in_batch[batch_vector]]
        features_for_conv2 = torch.cat([h_conv1_activated, root_features_to_concat_conv1], dim=1)
        features_for_conv2_dropped = F.dropout(features_for_conv2, p=self.dropout_rate, training=self.training)
        h_conv2 = self.conv2(features_for_conv2_dropped, edge_index, edge_weight=edge_weight)
        h_conv2_activated = F.leaky_relu(h_conv2)
        root_h_conv1_to_concat = h_conv1[root_indices_in_batch[batch_vector]]
        final_features_concat = torch.cat([h_conv2_activated, root_h_conv1_to_concat], dim=1)
        output_features = self.output_transform_linear(final_features_concat)
        output_features_activated = F.leaky_relu(output_features)
        return output_features_activated


# --- MAGNN-based Layer for Context Graph (HIN) (Placeholder) ---
class ContextGraphMAGNNLayer(Module):
    """
    Placeholder for a MAGNN (Metapath Aggregated Graph Neural Network) based layer for context graph (HIN).
    Actual implementation should include metapath instance encoding, intra-metapath attention aggregation,
    and inter-metapath attention aggregation.
    It should take a HeteroData object or related information as input.
    """

    def __init__(self, in_channels, out_channels, num_metapaths, num_heads, dropout_rate, **kwargs):
        super(ContextGraphMAGNNLayer, self).__init__()
        self.in_channels = in_channels  # Input feature dimension (dictionary by type or single dimension)
        self.out_channels = out_channels  # Output feature dimension
        self.num_metapaths = num_metapaths  # Number of metapaths to use
        self.num_heads = num_heads  # Number of attention heads between metapaths
        self.dropout_rate = dropout_rate

        # === Key components of MAGNN (conceptual) ===
        # 1. Metapath instance encoder (e.g., RNN or Transformer per metapath type)
        #    self.metapath_instance_encoders = nn.ModuleDict()
        #    for metapath_name in metapaths_definition: # Get metapath definitions from config, etc.
        #        self.metapath_instance_encoders[metapath_name] = SomeEncoder(...)

        # 2. Intra-metapath node aggregation (attention-based)
        #    self.intra_metapath_attentions = nn.ModuleDict()
        #    for metapath_name in metapaths_definition:
        #        self.intra_metapath_attentions[metapath_name] = AttentionMechanism(...)

        # 3. Inter-metapath aggregation (attention-based)
        #    self.inter_metapath_attention = MultiHeadAttention(out_channels, num_heads, ...)

        # Temporary: Simple linear transformation layer (needs to be replaced with actual MAGNN logic)
        # For HIN, input features can be a dictionary by node type, so logic to handle this is needed.
        # Here, it's assumed all node types have common in_channels and are input as a single tensor.
        if isinstance(in_channels, dict):
            # In reality, Linear layers per node type or a mapping process to a common space is needed.
            # This placeholder uses the dimension of the first node type for simplification.
            example_in_channel = list(in_channels.values())[0] if in_channels else 128  # Example
            self.placeholder_linear = nn.Linear(example_in_channel, out_channels)
            print(
                f"Warning: ContextGraphMAGNNLayer is a placeholder. Actual MAGNN logic is needed. Input channel (dictionary) processing: {in_channels}")
        else:
            self.placeholder_linear = nn.Linear(in_channels, out_channels)
            print(
                f"Warning: ContextGraphMAGNNLayer is a placeholder. Actual MAGNN logic is needed. Input channel (single): {in_channels}")

    def forward(self, x_context, data_object):
        """
        Args:
            x_context (Union[torch.Tensor, Dict[str, torch.Tensor]]):
                Node features of the context graph. Can be a single tensor or a dictionary of features by node type.
            data_object (torch_geometric.data.HeteroData or similar object):
                Object containing the entire structure information of the context graph (edges, metapath instances, etc.).
        Returns:
            torch.Tensor or Dict[str, torch.Tensor]: Updated node features.
        """
        # === Actual MAGNN logic (conceptual) ===
        # 1. Extract metapath instances from data_object
        #    metapath_instances = extract_metapath_instances(data_object, self.metapaths_definition)

        # 2. Encode each metapath instance and aggregate within metapath
        #    aggregated_features_per_metapath = {}
        #    for metapath_name, instances in metapath_instances.items():
        #        encoded_instances = self.metapath_instance_encoders[metapath_name](instances, x_context)
        #        # target_node_features are metapath-based features for a specific target node
        #        target_node_features = self.intra_metapath_attentions[metapath_name](encoded_instances, target_node_indices)
        #        aggregated_features_per_metapath[metapath_name] = target_node_features

        # 3. Inter-metapath attention aggregation (target node-centric)
        #    # For all target nodes, fuse aggregated features from various metapaths using attention again
        #    # Prepare tensor of shape (num_target_nodes, num_metapaths, out_channels)
        #    stacked_metapath_features = torch.stack(list(aggregated_features_per_metapath.values()), dim=1)
        #    final_node_embeddings = self.inter_metapath_attention(stacked_metapath_features) # (num_target_nodes, out_channels)

        # Temporary placeholder logic:
        if isinstance(x_context, dict):
            # Example of processing features by type (simply using features of the first type)
            # In reality, features of all types should be utilized, or features of a specific type should be used as main features
            first_node_type = list(x_context.keys())[0]
            x_to_transform = x_context[first_node_type]
            output_features = self.placeholder_linear(x_to_transform)
            # Result might also need to be returned as a dictionary by type
            return {first_node_type: F.leaky_relu(output_features)}
        elif x_context is not None and x_context.numel() > 0:  # Single tensor input
            output_features = self.placeholder_linear(x_context)
            return F.leaky_relu(output_features)
        else:  # If input is empty or None
            print("Warning: No valid input features for ContextGraphMAGNNLayer.")
            # Return empty or zero tensor of appropriate size
            # Output dimension and device should match other parts of the model
            # Example: target node count should be obtained from data_object
            num_target_nodes = x_context.size(0) if x_context is not None and x_context.numel() > 0 else 0
            return torch.zeros((num_target_nodes, self.out_channels),
                               device=self.placeholder_linear.weight.device if hasattr(self.placeholder_linear,
                                                                                       'weight') else 'cpu')


# --- Original KnowledgeGCNLayer (for reference or for homogeneous context graph) ---
class KnowledgeGCNLayer(Module):
    """
    (Legacy or simple homogeneous) GCN layer for knowledge/context graph.
    For HIN, it should be replaced with a specialized layer like ContextGraphMAGNNLayer.
    """

    def __init__(self, in_features, hidden_features, out_features, dropout_rate):
        super(KnowledgeGCNLayer, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(hidden_features, out_features, add_self_loops=False, normalize=False)
        self.output_transform_linear = nn.Linear(out_features, out_features)
        self.dropout_rate = dropout_rate

    def forward(self, x_node_features, edge_index, edge_weight=None):  # edge_weight added
        h_conv1 = self.conv1(x_node_features, edge_index, edge_weight=edge_weight)
        h_conv1_activated = F.leaky_relu(h_conv1)
        h_conv1_dropped = F.dropout(h_conv1_activated, p=self.dropout_rate, training=self.training)
        h_conv2 = self.conv2(h_conv1_dropped, edge_index, edge_weight=edge_weight)
        output_features = self.output_transform_linear(h_conv2)
        output_features_activated = F.leaky_relu(output_features)
        return output_features_activated


# --- Feature Fusion Layer ---
class TemporalFusionLayer(Module):
    """
    Fuses features from propagation graph, context graph, and an original feature set.
    This version assumes h_prop, h_context are outputs from GNNs, and h_original_common_nodes are initial text embeddings of common nodes.
    Updates h_prop and h_context for common nodes.
    """

    def __init__(self, feature_dim, bias=True):
        super(TemporalFusionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.weight_original = Parameter(torch.Tensor(feature_dim, feature_dim))
        self.weight_prop = Parameter(torch.Tensor(feature_dim, feature_dim))
        self.weight_context = Parameter(torch.Tensor(feature_dim, feature_dim))  # weight_k changed to weight_context
        self.fusion_linear = nn.Linear(feature_dim * 3, feature_dim)
        self.context_update_linear = nn.Linear(feature_dim * 2, feature_dim)

        self.bias = None
        if bias:
            self.bias = Parameter(torch.Tensor(feature_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.weight_original.data, gain=gain)
        nn.init.xavier_uniform_(self.weight_prop.data, gain=gain)
        nn.init.xavier_uniform_(self.weight_context.data, gain=gain)  # weight_k changed to weight_context
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.fusion_linear.weight, gain=gain)
        if self.fusion_linear.bias is not None: nn.init.zeros_(self.fusion_linear.bias)
        nn.init.xavier_uniform_(self.context_update_linear.weight, gain=gain)
        if self.context_update_linear.bias is not None: nn.init.zeros_(self.context_update_linear.bias)

    def forward(self, h_prop, h_context, h_original_common_nodes,
                common_node_indices_in_prop, common_node_indices_in_context):
        # h_prop: Node features from propagation graph GCN (num_prop_nodes, feature_dim)
        # h_context: Node features from context graph HIN-GNN (num_context_nodes, feature_dim)
        # h_original_common_nodes: Original text embeddings of common nodes (num_common_nodes, feature_dim)
        # common_node_indices_in_prop: Indices of common nodes within the h_prop tensor
        # common_node_indices_in_context: Indices of common nodes within the h_context tensor

        trans_orig_common = torch.mm(h_original_common_nodes, self.weight_original)
        trans_prop_common = torch.mm(h_prop[common_node_indices_in_prop], self.weight_prop)
        trans_context_common = torch.mm(h_context[common_node_indices_in_context],
                                        self.weight_context)  # h_k to h_context, weight_k to weight_context

        fused_common_features = torch.cat([trans_orig_common, trans_prop_common, trans_context_common], dim=1)
        fused_common_features = self.fusion_linear(fused_common_features)
        fused_common_activated = torch.tanh(fused_common_features)

        if self.bias is not None:
            fused_common_activated = fused_common_activated + self.bias

        h_prop_updated = h_prop.clone()
        h_context_updated = h_context.clone()  # h_k_updated changed to h_context_updated

        h_prop_updated[common_node_indices_in_prop] = fused_common_activated
        h_context_updated[
            common_node_indices_in_context] = fused_common_activated  # h_k_updated changed to h_context_updated

        return h_prop_updated, h_context_updated


# --- Global Mean Pooling Layer ---
class GlobalMeanPool(Module):
    """Performs global mean pooling over nodes in each graph in a batch."""

    def __init__(self):
        super(GlobalMeanPool, self).__init__()

    def forward(self, x_node_features, batch_vector):
        pooled_features = scatter_mean(x_node_features, batch_vector, dim=0)
        return pooled_features


# --- RNN Encoders (GRU/LSTM) ---
class GRUTextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(GRUTextEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional_factor = 2 if bidirectional else 1
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, packed_text_embeddings):
        _, hidden_state = self.gru(packed_text_embeddings)
        if self.bidirectional_factor == 2:
            last_hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        else:
            last_hidden = hidden_state[-1, :, :]
        return last_hidden


class BiLSTMTextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiLSTMTextEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bias=True, batch_first=True, bidirectional=True)

    def forward(self, packed_text_embeddings):
        _, (final_hidden_state, _) = self.lstm(packed_text_embeddings)
        concatenated_hidden = torch.cat((final_hidden_state[-2, :, :], final_hidden_state[-1, :, :]), dim=1)
        return concatenated_hidden


# --- Scaled Dot-Product Attention ---
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None, attention_mask=None):
        if scale is None:
            scale = K.size(-1) ** -0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores * scale
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, V)
        return context_vector, attention_weights
