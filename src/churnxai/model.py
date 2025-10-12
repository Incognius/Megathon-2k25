import torch
import torch.nn as nn
import numpy as np

class ChurnPredictor(nn.Module):
    """
    A PyTorch model for tabular data using entity embeddings for categorical features.
    """
    def __init__(self, preprocessor, embedding_dim_factor=0.5, hidden_layers=[100, 50], dropout_rate=0.4):
        """
        Args:
            preprocessor (ChurnDataPreprocessor): The fitted preprocessor, used to get vocab sizes.
            embedding_dim_factor (float): Factor to determine embedding size from vocab size.
            hidden_layers (list of int): List of neuron counts for hidden layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        
        # --- 1. Embedding Layers for Categorical Features ---
        self.embedding_layers = nn.ModuleList()
        cat_dims = [len(encoder.classes_) for encoder in preprocessor.categorical_encoders.values()]
        
        # Calculate embedding sizes (a common heuristic)
        self.embedding_sizes = [(c, min(50, int(c * embedding_dim_factor) + 1)) for c in cat_dims]
        
        for vocab_size, embedding_size in self.embedding_sizes:
            self.embedding_layers.append(nn.Embedding(vocab_size, embedding_size))
            
        total_embedding_size = sum(size for _, size in self.embedding_sizes)
        
        # --- 2. MLP for combined features ---
        num_numeric_features = len(preprocessor.numeric_cols)
        input_size = total_embedding_size + num_numeric_features
        
        layer_list = []
        for i, layer_size in enumerate(hidden_layers):
            if i == 0:
                layer_list.append(nn.Linear(input_size, layer_size))
            else:
                layer_list.append(nn.Linear(hidden_layers[i-1], layer_size))
            
            layer_list.append(nn.ReLU())
            layer_list.append(nn.BatchNorm1d(layer_size))
            layer_list.append(nn.Dropout(dropout_rate))
            
        self.layers = nn.Sequential(*layer_list)
        
        # --- 3. Output Layer ---
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        numeric_data = x['numeric']
        categorical_data = x['categorical']

        # Process embeddings
        embedded_features = []
        for i, layer in enumerate(self.embedding_layers):
            embedded_features.append(layer(categorical_data[:, i]))
        
        all_embeddings = torch.cat(embedded_features, 1)
        
        # Concatenate with numeric data
        combined_features = torch.cat([all_embeddings, numeric_data], 1)
        
        # Pass through MLP
        x = self.layers(combined_features)
        
        # Get final output
        output = self.output_layer(x)
        return output
