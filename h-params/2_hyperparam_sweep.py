import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer import TransformerC1 
from src.dataset import EncryptedDataset

def setup_ddp():
    """Initializes PyTorch Distributed Data Parallel environment."""
    # Disable P2P when mixing Turing (2080 Ti) and Ampere (3080) GPUs across PCIe roots
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return local_rank
    else:
        return 0 # Fallback to single GPU if not using torchrun

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def run_experiment(d_model, d_ff, n_heads, max_seq_len, num_steps, local_rank):
    """
    Trains the model with the given hyperparameters on a small subset of the data
    to verify convergence and capacity.
    """
    # 1. Initialize the dataset (subset of 200 lines for quick sweeps)
    dataset = EncryptedDataset('brown_plain.txt', 'brown_cipher.txt', subset_size=200, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 2. Initialize the model
    model = TransformerC1(
        vocab_size_plain=dataset.plain_vocab_size,
        vocab_size_cipher=dataset.cipher_vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
    ).to(local_rank)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    
    # 3. Dummy Training loop 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id_plain)
    
    model.train()
    step = 0
    while step < num_steps:
        for src, tgt in dataloader:
            if step >= num_steps:
                break
                
            src = src.to(local_rank)
            tgt = tgt.to(local_rank)
            
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], dataset.pad_id_cipher, dataset.pad_id_plain)
            
            output = output.contiguous().view(-1, output.size(-1))
            targets = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            if local_rank == 0 and (step % 50 == 0 or step == num_steps - 1):
                print(f"[{d_model=}, {d_ff=}, {n_heads=}] Step {step}/{num_steps} | Loss: {loss.item():.4f}")
            
            step += 1


    # if local_rank == 0:
    #     print(f"Finished sweep: d_model={d_model}, d_ff={d_ff}, heads={n_heads} | Final Loss: ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep C1 Hyperparameters")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1366)
    parser.add_argument("--num_steps", type=int, default=300)
    args = parser.parse_args()

    local_rank = setup_ddp()
    
    if local_rank == 0:
        print(f"Starting experiment on Ada cluster: d_model={args.d_model}, d_ff={args.d_ff}, n_heads={args.n_heads}, num_steps={args.num_steps}")

    try:
        run_experiment(args.d_model, args.d_ff, args.n_heads, args.max_seq_len, args.num_steps, local_rank)
    except Exception as e:
        print(f"Experiment failed (likely because src/models/ is not implemented yet!): {e}")
    finally:
        cleanup_ddp()

    # Note to user: Run this using torchrun to utilize multiple GPUs on the cluster:
    # torchrun --nproc_per_node=3 2_hyperparam_sweep.py --d_model=512 --d_ff=2048 --n_heads=8
