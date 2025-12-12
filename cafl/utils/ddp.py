import torch
import torch.distributed as dist

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if ddp_is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def global_num_pos(num_pos_local: int) -> int:
    t = torch.tensor([num_pos_local], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.long)
    t = all_reduce_sum(t)
    return int(t.item())
