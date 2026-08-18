import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from huggingface_hub import HfApi

# Import components
from src.dataset import EncryptedDataset, TokenFreeDataset
from src.models.attention import MultiHeadAttention, GroupedQueryAttention
from src.models.norm import LayerNorm, RMSNorm
from src.models.positional import SinusoidalPositionalEncoding, get_rotary_matrix
from src.utils import greedy_decode, bit_level_accuracy, sequence_accuracy, calculate_levenshtein, calculate_bleu, calculate_rouge
from src.models.blt import BLTTransformer

# -----------------------------------------------------------------------
# Core Transformer Assembly (C1, C2, C3, C4)
# -----------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, attention_type='mha', norm_type='layernorm'):
        super().__init__()
        if attention_type == 'mha':
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == 'gqa':
            # C3 configuration: say 8 query heads, 2 kv heads
            self.self_attn = GroupedQueryAttention(d_model, n_heads, max(1, n_heads // 4), dropout)
            
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        NormClass = LayerNorm if norm_type == 'layernorm' else RMSNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, sin=None, cos=None):
        # Pre-LayerNorm configuration
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, mask=mask, sin=sin, cos=cos)
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, attention_type='mha', norm_type='layernorm'):
        super().__init__()
        if attention_type == 'mha':
            self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == 'gqa':
            self.self_attn = GroupedQueryAttention(d_model, n_heads, max(1, n_heads // 4), dropout)
            self.cross_attn = GroupedQueryAttention(d_model, n_heads, max(1, n_heads // 4), dropout)
            
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        NormClass = LayerNorm if norm_type == 'layernorm' else RMSNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        self.norm3 = NormClass(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, sin=None, cos=None):
        # Self Attention (masked)
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask, sin=sin, cos=cos)
        x = x + self.dropout(attn_out)
        
        # Cross Attention (no RoPE for cross-attention usually, but we pass sin/cos if configured, though standard is self-attn only)
        # Note: We only apply RoPE to cross-attention if we want to, but standard RoPE is for self-attention.
        # We will not apply RoPE to cross-attention to strictly follow standard.
        attn_out = self.cross_attn(self.norm2(x), enc_output, enc_output, mask=src_mask)
        x = x + self.dropout(attn_out)
        
        # Feed Forward
        ff_out = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_out)
        return x

class Transformer(nn.Module):
    """
    Generalized Transformer to support C1, C2, C3, C4.
    """
    def __init__(self, vocab_size_plain, vocab_size_cipher, d_model=512, n_heads=8, d_ff=2048, 
                 num_layers=6, max_seq_len=1366, dropout=0.1, config='C1'):
        super().__init__()
        self.config = config
        self.d_model = d_model
        
        self.src_emb = nn.Embedding(vocab_size_cipher, d_model)
        self.tgt_emb = nn.Embedding(vocab_size_plain, d_model)
        
        attention_type = 'gqa' if config == 'C3' else 'mha'
        norm_type = 'rmsnorm' if config == 'C4' else 'layernorm'
        self.pos_enc_type = 'rope' if config == 'C2' else 'sinusoidal'
        
        if self.pos_enc_type == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
            
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, attention_type, norm_type) for _ in range(num_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, attention_type, norm_type) for _ in range(num_layers)
        ])
        
        NormClass = LayerNorm if norm_type == 'layernorm' else RMSNorm
        self.final_norm = NormClass(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size_plain)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt, pad_idx_src, pad_idx_tgt):
        src_mask = (src == pad_idx_src).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt == pad_idx_tgt).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_sub_mask = ~tgt_sub_mask.unsqueeze(0).unsqueeze(1)
        tgt_mask = tgt_pad_mask | tgt_sub_mask
        return src_mask, tgt_mask
        
    def forward(self, src, tgt, pad_idx_src, pad_idx_tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt, pad_idx_src, pad_idx_tgt)
        
        enc_out = self.src_emb(src)
        dec_out = self.tgt_emb(tgt)
        
        sin, cos = None, None
        if self.pos_enc_type == 'sinusoidal':
            enc_out = self.pos_enc(enc_out)
            dec_out = self.pos_enc(dec_out)
        else: # RoPE
            # For RoPE, we generate sin and cos for the sequence lengths
            seq_len = src.size(1)
            # Assuming n_heads = 8 for RoPE dimension
            d_k = self.d_model // 8 
            sin, cos = get_rotary_matrix(max(src.size(1), tgt.size(1)), d_k, src.device)
            
        enc_out = self.dropout(enc_out)
        for block in self.encoder_blocks:
            enc_out = block(enc_out, mask=src_mask, sin=sin[:src.size(1)] if sin is not None else None, cos=cos[:src.size(1)] if cos is not None else None)
        enc_out = self.final_norm(enc_out) 
            
        dec_out = self.dropout(dec_out)
        for block in self.decoder_blocks:
            dec_out = block(dec_out, enc_out, src_mask, tgt_mask, sin=sin[:tgt.size(1)] if sin is not None else None, cos=cos[:tgt.size(1)] if cos is not None else None)
            
        dec_out = self.final_norm(dec_out)
        out = self.fc_out(dec_out)
        return out


# -----------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------
def setup_ddp():
    os.environ["NCCL_P2P_DISABLE"] = "1"
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return local_rank
    return 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def evaluate(model, val_loader, dataset, device, local_rank):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id_plain)
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            if hasattr(model, 'module'):
                output = model.module(src, tgt[:, :-1], dataset.pad_id_cipher, dataset.pad_id_plain)
            else:
                output = model(src, tgt[:, :-1], dataset.pad_id_cipher, dataset.pad_id_plain)
                
            output = output.contiguous().view(-1, output.size(-1))
            targets = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, targets)
            total_loss += loss.item()
            
    # Compute metrics using greedy decode on a small subset
    # TODO: generate outputs and calculate BLEU, ROUGE, Levenshtein, Bit/Seq accuracy
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Variants")
    parser.add_argument("--config", type=str, default="C1", choices=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        wandb.init(project="anlp-assignment-1", name=f"run-{args.config}", config=args)

    # 1. Dataset
    is_blt = args.config == "C5"
    if is_blt:
        dataset = TokenFreeDataset('brown_plain.txt', 'brown_cipher.txt', max_seq_len=2048)
    else:
        dataset = EncryptedDataset('brown_plain.txt', 'brown_cipher.txt', max_seq_len=1366)
        
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None))

    # 2. Model
    if is_blt:
        model = BLTTransformer(
            vocab_size_plain=dataset.plain_vocab_size,
            d_model=args.d_model,
            patch_size=4
        ).to(device)
    else:
        model = Transformer(
            vocab_size_plain=dataset.plain_vocab_size,
            vocab_size_cipher=dataset.cipher_vocab_size,
            d_model=args.d_model,
            config=args.config
        ).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id_plain)

    # 4. Training Loop
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        for step, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            if is_blt:
                output = model(src, tgt[:, :-1], dataset.pad_id_plain)
            else:
                output = model(src, tgt[:, :-1], dataset.pad_id_cipher, dataset.pad_id_plain)
            
            output = output.contiguous().view(-1, output.size(-1))
            targets = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if local_rank == 0 and step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

        # Save checkpoint at the end of epoch
        if local_rank == 0:
            torch.save(model.state_dict(), f"outputs/checkpoint_{args.config}_epoch_{epoch}.pt")
            
    if local_rank == 0:
        if args.push_to_hub and args.hf_token:
            api = HfApi()
            # Push to hub logic here
            print("Pushed to Hugging Face Hub")
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    main()
