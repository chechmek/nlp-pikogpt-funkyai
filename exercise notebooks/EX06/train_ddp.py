import os
import time
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Same toy setup style as EX04, so only the training orchestration changes.
class ToyNextTokenDataset(Dataset):
    def __init__(self, num_samples=256, seq_len=64, vocab_size=100, pad_token_id=0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def __len__(self):
        # Number of training examples in this dataset.
        return self.num_samples

    def __getitem__(self, idx):
        # Random token IDs in [1, vocab_size-1]. Token 0 is reserved for padding.
        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)

        # Simulate variable-length sequences to demonstrate padding effects.
        effective_len = torch.randint(low=self.seq_len // 2, high=self.seq_len + 1, size=(1,)).item()
        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        padding_mask[:effective_len] = True  # True = real token, False = padding
        input_ids[~padding_mask] = self.pad_token_id

        # Labels are next-token targets; padded positions are ignored with -100.
        labels = input_ids.clone()
        labels[~padding_mask] = -100
        return {"input_ids": input_ids, "labels": labels, "padding_mask": padding_mask}


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)  # [B,S] -> [B,S,H]
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),  # [B,S,H] -> [B,S,V]
        )

    def forward(self, input_ids, padding_mask=None):
        # padding_mask is passed to mirror a real LM API.
        x = self.embed(input_ids)
        return self.ff(x)

def setup_distributed():
    """Initialize process group and return rank info + device (CPU or GPU)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Provided by torchrun.
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"  # NCCL for NVIDIA GPUs, Gloo for CPU fallback.
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()  # Global process id (0 ... world_size-1).
    world_size = dist.get_world_size()  # Total number of processes.

    if use_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)  # Ensure this process only uses its assigned GPU.
    else:
        device = torch.device("cpu")

    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} device={device}",
        flush=True,
    )
    return rank, world_size, local_rank, device


def cleanup_distributed():
    # Clean shutdown so repeated runs do not leave hanging process groups.
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch_ddp(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_accum_steps=4,
    log_every=20,
):
    model.train()  # Enable train behavior (e.g., dropout).
    running_loss = 0.0
    epoch_tokens_local = 0
    optimizer_updates = 0
    optimizer.zero_grad(set_to_none=True)  # Start with empty gradient buffers.
    token_count_window = 0  # Number of valid tokens since last optimizer step.
    world_size = dist.get_world_size()

    for step, batch in enumerate(loader, start=1):
        # Move this micro-batch to the local GPU.
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True).bool()

        # Forward pass on local GPU.
        logits = model(input_ids, padding_mask=padding_mask)

        # Shift for next-token prediction: predict token t+1 from positions <= t.
        logits_shift = logits[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()

        vocab_size = logits_shift.size(-1)
        # IMPORTANT: sum (not mean) so token weighting is handled globally, not per micro-batch.
        loss = F.cross_entropy(
            logits_shift.view(-1, vocab_size),
            labels_shift.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        # Count valid targets (non-padding) for this micro-batch.
        valid_tokens = (labels_shift != -100).sum().item()
        epoch_tokens_local += valid_tokens
        token_count_window += valid_tokens

        # Sync gradients only on update steps (or at epoch end for leftover micro-batches).
        should_sync = (step % grad_accum_steps == 0) or (step == len(loader))
        if should_sync:
            # On sync step, DDP performs all-reduce during backward.
            loss.backward()
        else:
            # On non-sync steps, accumulate local grads only (faster, less communication).
            with model.no_sync():
                loss.backward()

        if should_sync:
            local_tokens_window = token_count_window
            # Total valid tokens across ALL GPUs and all micro-batches in this accumulation window.
            token_tensor = torch.tensor(token_count_window, device=device, dtype=torch.float32)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            total_tokens_global = token_tensor.item()

            # DDP gives averaged grads across ranks after sync.
            # Multiply by world_size to recover summed grads, then divide by global token count.
            grad_scale = world_size / max(total_tokens_global, 1.0)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(grad_scale)

            optimizer.step()  # Apply one update for the full accumulation window.
            scheduler.step()  # Keep LR schedule aligned with optimizer steps.
            optimizer_updates += 1
            update_step = (step + grad_accum_steps - 1) // grad_accum_steps
            if dist.get_rank() == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    "[sync] "
                    f"micro_step={step:04d} update_step={update_step:04d} "
                    f"local_tokens_window={local_tokens_window} "
                    f"global_tokens_window={int(total_tokens_global)} "
                    f"grad_scale={grad_scale:.6e} lr={lr:.2e}",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)  # Reset buffers for next window.
            token_count_window = 0

        # Logging metric only: average token loss for this micro-batch.
        running_loss += loss.item() / max(valid_tokens, 1)

        if step % log_every == 0 and dist.get_rank() == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"step={step:04d} avg_micro_loss={running_loss/step:.4f} lr={lr:.2e}")

    epoch_tokens_tensor = torch.tensor(epoch_tokens_local, device=device, dtype=torch.float64)
    dist.all_reduce(epoch_tokens_tensor, op=dist.ReduceOp.SUM)
    epoch_tokens_global = int(epoch_tokens_tensor.item())
    return {
        "train_loss_epoch": running_loss / len(loader),
        "epoch_tokens_global": epoch_tokens_global,
        "optimizer_updates": optimizer_updates,
    }

def main_ddp():
    # Every spawned process runs this function once.
    rank, world_size, local_rank, device = setup_distributed()
    torch.manual_seed(7 + rank)  # Different seed per rank avoids identical random batches.

    try:
        dataset = ToyNextTokenDataset(num_samples=1024, seq_len=64, vocab_size=100)
        # DistributedSampler splits dataset indices across ranks (no overlap within epoch).
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=8,  # Per-GPU batch size (micro-batch size).
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,  # Keeps micro-batch shapes consistent.
        )

        model = TinyLanguageModel(vocab_size=100, hidden_dim=128).to(device)
        # Wrap model so gradient synchronization happens automatically on backward sync steps.
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

        epochs = 2
        grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
        updates_per_epoch = math.ceil(len(loader) / grad_accum_steps)
        total_steps = updates_per_epoch * epochs  # Scheduler should track optimizer updates, not micro-steps.
        optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda s: 1.0 - 0.8 * (s / max(total_steps, 1)))
        per_gpu_batch = 8
        if rank == 0:
            eff_batch = per_gpu_batch * world_size * grad_accum_steps
            print(
                f"[config] world_size={world_size} per_gpu_batch={per_gpu_batch} "
                f"grad_accum_steps={grad_accum_steps} effective_batch={eff_batch} "
                "(override with GRAD_ACCUM_STEPS=<int>)",
                flush=True,
            )
        for epoch in range(epochs):
            # Needed when shuffle=True so each epoch gets a fresh (but rank-consistent) shuffle.
            sampler.set_epoch(epoch)
            epoch_start = time.perf_counter()
            metrics = train_one_epoch_ddp(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_accum_steps=grad_accum_steps,
                log_every=20,
            )
            epoch_time_s = time.perf_counter() - epoch_start
            if rank == 0:
                # Effective batch seen by one optimizer step across all GPUs.
                eff_batch = per_gpu_batch * world_size * grad_accum_steps
                tok_per_s = metrics["epoch_tokens_global"] / max(epoch_time_s, 1e-9)
                print(
                    f"[perf] epoch={epoch} epoch_time_s={epoch_time_s:.2f} "
                    f"tokens_per_s={tok_per_s:.1f} optimizer_updates={metrics['optimizer_updates']}",
                    flush=True,
                )
                print(f"epoch={epoch} metrics={metrics} effective_batch={eff_batch}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main_ddp()
    
# Important:
# Run with torchrun from terminal
# torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ddp.py
