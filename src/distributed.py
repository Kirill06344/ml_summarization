"""Минимальная обёртка над torch.distributed: один процесс на GPU, код обучения не ветвится на «один или много»."""
from __future__ import annotations

import contextlib
import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Context:
    rank: int
    world: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def gather(self, value) -> list:
        return dist_gather(value) if self.world > 1 else [value]

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def all_true(self, value: bool) -> bool:
        """Согласованное решение: если процессы разойдутся в выборе пропустить батч, обучение зависнет."""
        if self.world == 1:
            return value
        flag = torch.tensor(int(value), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def any_true(self, value: bool) -> bool:
        return not self.all_true(not value)

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.world == 1:
            return module
        device_ids = [self.device.index] if self.device.type == 'cuda' else None
        return torch.nn.parallel.DistributedDataParallel(module, device_ids=device_ids)

    def no_sync(self, module: torch.nn.Module, skip: bool):
        return module.no_sync() if self.world > 1 and skip else contextlib.nullcontext()


def dist_gather(value) -> list:
    output = [None] * dist.get_world_size()
    dist.all_gather_object(output, value)
    return output


def setup(rank: int, world: int) -> Context:
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda', rank) if use_gpu else torch.device('cpu')
    if use_gpu:
        torch.cuda.set_device(device)
    if world > 1:
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29517')
        # Генерация на val занимает десятки минут, а коллективные операции по умолчанию ждут 30.
        dist.init_process_group('nccl' if use_gpu else 'gloo', rank=rank, world_size=world,
                                timeout=datetime.timedelta(hours=3))
    return Context(rank, world, device)


def teardown(context: Context) -> None:
    if context.world > 1:
        dist.destroy_process_group()
