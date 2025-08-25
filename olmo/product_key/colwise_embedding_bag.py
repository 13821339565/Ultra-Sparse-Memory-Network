# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
    Partial,
)

from torch.distributed.tensor.parallel.style import ParallelStyle

from .xformer_embeddingbag import xformers_embedding_bag


import logging

logger = logging.getLogger()


class xFormerEmbeddingBag(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, dim, dtype=torch.bfloat16))

    def forward(self, indices, scores):
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
            num_shards = self.weight.device_mesh.size()
            if num_shards > 1:
                # scale gradients so that we end up with the average rather than sum
                grad_scale = 1 / num_shards
                weight = weight * grad_scale + (weight * (1-grad_scale)).detach()
        else:
            weight = self.weight
        # output = F.embedding_bag(indices, weight, per_sample_weights=scores, mode="sum")
        output = xformers_embedding_bag(
            indices, weight, per_sample_weights=scores, mode="sum"
        )
        return output


class ColwiseEmbeddingBag(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts= None,
        output_layouts= None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(0),)
        self.output_layouts = (output_layouts or Shard(0),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # annotate module input placements/sharding with input_layouts
        dist_inputs = tuple()
        for t in inputs:
            if t is None:
                dist_inputs += (None,)
                continue
            input_tensor = t
            if not isinstance(input_tensor, DTensor):
                input_tensor = DTensor.from_local(
                    input_tensor, device_mesh, input_layouts, run_check=False
                )

            # transform the input layouts to the desired layouts of ColwiseEmbeddingBag
            if input_layouts != desired_input_layouts:
                input_tensor = input_tensor.redistribute(
                    placements=desired_input_layouts, async_op=False
                )
            dist_inputs += (input_tensor.to_local(grad_placements=(Partial(),)),)
        return dist_inputs

    def _partition_embeddingbag_fn(self, name, module, device_mesh):
        # Only column parallelize the weights of EmbeddingBag
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(1)]))
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        outputs = DTensor.from_local(
            outputs, device_mesh, (Shard(-1),), run_check=False
        )
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=False)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, xFormerEmbeddingBag):
            partition_fn = self._partition_embeddingbag_fn
        else:
            raise NotImplementedError(
                "ColwiseEmbeddingBag currently only support nn.EmbeddingBag!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )