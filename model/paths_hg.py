import os
import sys

class PathSet:
    """
    Manages paths for datasets and models.
    """

    def __init__(self, dataset_name, base_data_dir="/home//data"):
        """
        Initializes paths for a given dataset.
        Args:
            dataset_name (str): Dataset name.
            base_data_dir (str): The root directory where all datasets are stored.
        """
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(base_data_dir, dataset_name)

        if not os.path.exists(self.data_dir) and dataset_name:  # Check only if dataset_name is not empty
            print(f"Warning: Dataset directory not found at {self.data_dir}")
        self.node_to_idx_mid_path = os.path.join(self.data_dir, "node2idx_mid.txt")
        self.mid_to_text_path = os.path.join(self.data_dir, 'mid2text.txt')

        # Propagation graph data paths
        self.propagation_data_dir = os.path.join(self.data_dir, f"{dataset_name}_propagation_data")
        if not os.path.exists(self.propagation_data_dir) and dataset_name:
            print(f"Warning: Propagation graph data directory not found at {self.propagation_data_dir}")

        self.propagation_node_idx_path = os.path.join(self.propagation_data_dir, "propagation_node_idx.npy")
        self.propagation_node_graph_path = os.path.join(self.propagation_data_dir,
                                                        "propagation_adj.npy")  # adj or edge_index format
        self.propagation_label_path = os.path.join(self.propagation_data_dir, "label.npy")
        self.propagation_root_idx_path = os.path.join(self.propagation_data_dir, "propagation_root_index.npy")

        # Context graph (HIN) data paths (newly added or modified)
        self.context_hin_data_dir = os.path.join(self.data_dir, f"{dataset_name}_context_hin_data")
        if not os.path.exists(self.context_hin_data_dir) and dataset_name:
            print(f"Warning: Context graph (HIN) data directory not found at {self.context_hin_data_dir}")

        self.saved_model_dir = "trained_models_output/"  # Consistently modified save path
        if not os.path.exists(self.saved_model_dir):
            os.makedirs(self.saved_model_dir, exist_ok=True)


class PathSetBERT(PathSet):


    def __init__(self, dataset_name, base_data_dir="/home//data",
                 bert_base_dir="bert_models/"):  # Changed default BERT base directory
        super().__init__(dataset_name, base_data_dir)

       if dataset_name == 'weibo':
            self.bert_model_name_or_path = os.path.join(bert_base_dir, 'klue/bert-base')
            self.bert_vocab_file = 'vocab.txt'  # Typically part of the pre-trained model directory
        elif dataset_name == 'twitter':
            self.bert_model_name_or_path = os.path.join(bert_base_dir, 'bert-base-uncased')
            self.bert_vocab_file = 'vocab.txt'
        else:
            print(f"Warning: BERT model path not configured for dataset: {dataset_name}")
            self.bert_model_name_or_path = None
            self.bert_vocab_file = None

        if self.bert_model_name_or_path and not os.path.exists(self.bert_model_name_or_path):
            print(f"Warning: BERT model directory not found at {self.bert_model_name_or_path}")


class PathSetEarlyDetection(PathSetBERT):  # Renamed from PathSet_early
    """
    Manages paths for early detection datasets, which might have multiple time-sliced versions.
    """

    def __init__(self, dataset_name, time_counts,
                 base_data_dir="/home//data",
                 bert_base_dir="bert_models/"):

        super().__init__(dataset_name, base_data_dir, bert_base_dir)
        self.time_counts = time_counts

        self.early_train_paths_map = {}
        self.early_test_paths_map = {}
        self._initialize_early_detection_paths()

    def _initialize_early_detection_paths(self):
        """Helper method to populate the path maps."""
        path_keys_propagation = [
            "propagation_node_idx", "propagation_adj",

            "label", "propagation_root_index"
        ]

        for time_slice, count_slice in self.time_counts:
            prop_folder_name = f"{self.dataset_name}_propagation_data_time{time_slice}_count{count_slice}"
            current_early_prop_dir = os.path.join(self.data_dir, prop_folder_name)
            if not os.path.exists(current_early_prop_dir):
                print(f"Warning: Early detection propagation data directory not found: {current_early_prop_dir}")

            train_paths_prop_slice = {}
            test_paths_prop_slice = {}
            for key in path_keys_propagation:
                train_paths_prop_slice[key] = os.path.join(current_early_prop_dir, f"{key}_train.npy")
                test_paths_prop_slice[key] = os.path.join(current_early_prop_dir, f"{key}_test.npy")

            self.early_train_paths_map[('propagation', time_slice, count_slice)] = train_paths_prop_slice
            self.early_test_paths_map[('propagation', time_slice, count_slice)] = test_paths_prop_slice

    def get_early_detection_paths_for_slice(self, graph_type, time_slice, count_slice, split_type="train"):
        """
        Retrieves a dictionary of paths for a specific time/count slice and split type.
        Args:
            graph_type (str): 'propagation' or 'context'.
            time_slice (int): The time slice identifier.
            count_slice (int): The count slice identifier.
            split_type (str): 'train' or 'test'.
        Returns:
            dict: A dictionary of paths for the specified slice and split, or None if not found.
        """
        key = (graph_type, time_slice, count_slice)
        path_map = self.early_train_paths_map if split_type == "train" else self.early_test_paths_map

        if key not in path_map:
            print(
                f"Warning: No paths found for slice ({time_slice}, {count_slice}), graph type '{graph_type}', and split '{split_type}'.")
            return None
        return path_map[key]
