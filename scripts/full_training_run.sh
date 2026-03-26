# example script
#
# ── Single-GPU (default) ────────────────────────────────────
# python main.py --stage train --config configs/train_default.toml
# python main.py --stage train --config configs/train_large.toml
#
# ── Multi-GPU with DDP (torchrun) ───────────────────────────
# torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     main.py --stage train --config configs/train_large.toml
#
# Adjust --nproc_per_node to the number of GPUs available.
# The effective batch size equals:
#   batch_size × nproc_per_node × grad_accum_steps