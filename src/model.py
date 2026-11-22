from torch import Tensor
from typing import override
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        hidden_layer_size: int,
        input_size: int,
        output_size: int,
        num_layers: int,
        dropout_rate: float,
    ) -> None:
        
        super().__init__()

        self.dropout: nn.Dropout = nn.Dropout(dropout_rate)

        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_size,
            padding_idx=0,
        )

        self.model: nn.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.linear: nn.Linear = nn.Linear(
            in_features=hidden_layer_size * 2,
            out_features=output_size,
        )


    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embed_x: torch.Tensor = self.dropout(self.embedding(x))

        output, (hidden, cell) = self.model(embed_x)

        if self.model.bidirectional:
            hidden: Tensor = torch.cat(tensors=(hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.linear(self.dropout(hidden))