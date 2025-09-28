from __future__ import annotations

from typing import Dict, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import IterableDataset as HFIterableDataset

from policy_index import policy_index


EXPECTED_SEQ_LEN = 72
EXTRA_TOKENS = 2


class ChessPolicyDataset(IterableDataset):
    """Streaming-only dataset wrapper that validates incoming examples."""

    def __init__(
        self,
        hf_dataset: HFIterableDataset,
        act_token_id: int,
        think_token_id: int,
    ) -> None:
        if act_token_id is None:
            raise ValueError("act_token_id must be provided")
        if think_token_id is None:
            raise ValueError("think_token_id must be provided")
        if not isinstance(hf_dataset, HFIterableDataset):
            raise TypeError("ChessPolicyDataset requires a streaming Hugging Face dataset")

        self.dataset = hf_dataset
        self.act_token_id = int(act_token_id)
        self.think_token_id = int(think_token_id)
        self.base_seq_len = EXPECTED_SEQ_LEN - EXTRA_TOKENS
        self.policy_size = len(policy_index)

    @property
    def is_streaming(self) -> bool:
        return True

    def __len__(self) -> int:
        raise TypeError("ChessPolicyDataset wraps streaming data and has no length")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is not None:
            shard = self.dataset.shard(worker_info.num_workers, worker_info.id)
            iterator: Iterable[Dict[str, object]] = iter(shard)
        else:
            iterator = iter(self.dataset)

        for example in iterator:
            yield self._process_example(example)

    def _process_example(self, example: Dict[str, object]) -> Dict[str, torch.Tensor]:
        policy = torch.as_tensor(example["policy"])
        if policy.dtype != torch.float32:
            policy = policy.to(dtype=torch.float32)

        if policy.ndim != 1 or policy.shape[0] != self.policy_size:
            raise ValueError(
                f"policy field expected shape ({self.policy_size},), received {tuple(policy.shape)}"
            )

        input_ids = torch.as_tensor(example["input_ids"])
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)

        if input_ids.ndim != 1:
            raise ValueError(
                f"input_ids expected 1D tensor, received shape {tuple(input_ids.shape)}"
            )

        seq_len = input_ids.shape[0]
        if seq_len == EXPECTED_SEQ_LEN:
            if input_ids[-2].item() != self.think_token_id or input_ids[-1].item() != self.act_token_id:
                raise ValueError(
                    "input_ids final tokens do not match expected <THINK> and <ACT> token ids"
                )
        elif seq_len == EXPECTED_SEQ_LEN - 1:
            if input_ids[-1].item() != self.act_token_id:
                raise ValueError(
                    f"input_ids final token id {input_ids[-1].item()} does not match expected act token id {self.act_token_id}"
                )
            think_token = input_ids.new_tensor([self.think_token_id])
            act_token = input_ids.new_tensor([self.act_token_id])
            input_ids = torch.cat((input_ids[:-1], think_token, act_token))
        elif seq_len == EXPECTED_SEQ_LEN - EXTRA_TOKENS:
            think_token = input_ids.new_tensor([self.think_token_id])
            act_token = input_ids.new_tensor([self.act_token_id])
            input_ids = torch.cat((input_ids, think_token, act_token))
        else:
            raise ValueError(
                f"input_ids length {seq_len} does not match expected {EXPECTED_SEQ_LEN}"
            )

        original_seq_len = input_ids.shape[0] - EXTRA_TOKENS
        causal_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        causal_mask[original_seq_len:] = True

        wdl = torch.as_tensor(example["wdl"])
        if wdl.dtype != torch.float32:
            wdl = wdl.to(dtype=torch.float32)

        if wdl.ndim != 1 or wdl.shape[0] != 3:
            raise ValueError(
                f"wdl field expected shape (3,), received {tuple(wdl.shape)}"
            )

        return {
            "policy": policy,
            "input_ids": input_ids,
            "causal_mask": causal_mask,
            "wdl": wdl,
        }


class ChessPolicyCollator:
    """Collator that batches tensor creation for already-tokenized data."""

    def __init__(self) -> None:
        self.policy_size = len(policy_index)

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        policy_list = []
        wdl_list = []
        causal_masks = []
        for item in batch:
            input_ids_list.append(item["input_ids"])
            policy_list.append(item["policy"])
            wdl_list.append(item["wdl"])
            causal_masks.append(item["causal_mask"])

        if not input_ids_list:
            raise ValueError("Empty batch provided to ChessPolicyCollator")

        input_ids = torch.stack(input_ids_list)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)

        if input_ids.ndim != 2 or input_ids.shape[1] != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Batch tokenized length {input_ids.shape[1]} does not match expected {EXPECTED_SEQ_LEN}"
            )

        causal_mask_tensor = torch.stack(causal_masks)
        if causal_mask_tensor.ndim != 2 or causal_mask_tensor.shape[1] != EXPECTED_SEQ_LEN:
            raise ValueError(
                "causal mask shape mismatch after batching"
            )

        policy_values = torch.stack(policy_list)
        if policy_values.dtype != torch.float32:
            policy_values = policy_values.to(dtype=torch.float32)
        if policy_values.shape[1] != self.policy_size:
            raise ValueError(
                f"policy tensor expected width {self.policy_size}, "
                f"received {policy_values.shape[1]}"
            )

        wdl_values = torch.stack(wdl_list)
        if wdl_values.dtype != torch.float32:
            wdl_values = wdl_values.to(dtype=torch.float32)
        if wdl_values.shape[1] != 3:
            raise ValueError(
                f"wdl tensor expected width 3, received {wdl_values.shape[1]}"
            )

        return {
            "input_ids": input_ids,
            "policy": policy_values,
            "wdl": wdl_values,
            "causal_mask": causal_mask_tensor,
        }


def create_dataloader(
    hf_dataset: HFIterableDataset,
    act_token_id: int,
    think_token_id: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = ChessPolicyDataset(
        hf_dataset,
        act_token_id=act_token_id,
        think_token_id=think_token_id,
    )
    collator = ChessPolicyCollator()

    if shuffle:
        raise ValueError(
            "Shuffling must be applied to the source dataset before wrapping in ChessPolicyDataset"
        )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
    )
