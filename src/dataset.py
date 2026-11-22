from typing import override
import numpy as np
import torch
from torch.utils.data import Dataset
from .process_dataset import tokens_to_ids, pad_or_truncate


class BookDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __init__(
        self,
        tokens: list[list[str]],
        labels: np.ndarray,
        vocab:  dict[str, int],
        max_length: int,
    ) -> None:
        
        self.tokens: list[list[str]] = tokens
        self.vocab: dict[str, int] = vocab
        self.max_length: int = max_length
        self.labels: torch.Tensor = torch.tensor(data=labels, dtype=torch.long)

    
    def __len__(self) -> int:
        return len(self.labels)


    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        ids: list[int] = tokens_to_ids(tokens=self.tokens[index], vocab=self.vocab)
        ids = pad_or_truncate(ids, self.max_length)

        ids_tensor: torch.Tensor = torch.tensor(data=ids, dtype=torch.long)
        
        return ids_tensor, self.labels[index]
