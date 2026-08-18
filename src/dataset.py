import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class EncryptedDataset(Dataset):
    def __init__(self, plain_file, cipher_file, max_seq_len=1366, subset_size=None):
        # Robust relative paths: Resolve relative to this file's location
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        plain_path = os.path.join(base_dir, 'Dataset_A1', plain_file)
        cipher_path = os.path.join(base_dir, 'Dataset_A1', cipher_file)
        
        plain_tok_path = os.path.join(base_dir, 'h-params', 'tokenizers', 'plain_tokenizer.json')
        cipher_tok_path = os.path.join(base_dir, 'h-params', 'tokenizers', 'cipher_tokenizer.json')
        
        if not os.path.exists(plain_tok_path) or not os.path.exists(cipher_tok_path):
            raise FileNotFoundError("Tokenizers not found. Please run h-params/1_max_seq_len.py first.")
            
        self.plain_tokenizer = Tokenizer.from_file(plain_tok_path)
        self.cipher_tokenizer = Tokenizer.from_file(cipher_tok_path)
        
        self.max_seq_len = max_seq_len
        
        with open(plain_path, 'r', encoding='utf-8') as f:
            self.plain_lines = f.readlines()
            
        with open(cipher_path, 'r', encoding='utf-8') as f:
            self.cipher_lines = f.readlines()
            
        if subset_size is not None:
            self.plain_lines = self.plain_lines[:subset_size]
            self.cipher_lines = self.cipher_lines[:subset_size]
            
        assert len(self.plain_lines) == len(self.cipher_lines), "Mismatched line counts!"
        
        self.plain_vocab_size = self.plain_tokenizer.get_vocab_size()
        self.cipher_vocab_size = self.cipher_tokenizer.get_vocab_size()
        
        self.bos_id_plain = self.plain_tokenizer.token_to_id("[BOS]")
        self.eos_id_plain = self.plain_tokenizer.token_to_id("[EOS]")
        self.pad_id_plain = self.plain_tokenizer.token_to_id("[PAD]")
        
        self.bos_id_cipher = self.cipher_tokenizer.token_to_id("[BOS]")
        self.eos_id_cipher = self.cipher_tokenizer.token_to_id("[EOS]")
        self.pad_id_cipher = self.cipher_tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.plain_lines)

    def _pad_sequence(self, token_ids, bos_id, eos_id, pad_id):
        # Truncate if too long (leaving room for BOS/EOS)
        token_ids = token_ids[:self.max_seq_len - 2]
        
        # Add BOS and EOS
        sequence = [bos_id] + token_ids + [eos_id]
        
        # Pad up to max_seq_len
        padding_length = self.max_seq_len - len(sequence)
        sequence = sequence + [pad_id] * padding_length
        
        return torch.tensor(sequence, dtype=torch.long)

    def __getitem__(self, idx):
        plain_line = self.plain_lines[idx].strip()
        cipher_line = self.cipher_lines[idx].strip()
        
        # Tokenize
        plain_encoded = self.plain_tokenizer.encode(plain_line).ids
        cipher_encoded = self.cipher_tokenizer.encode(cipher_line).ids
        
        # Pad & add special tokens
        plain_tensor = self._pad_sequence(plain_encoded, self.bos_id_plain, self.eos_id_plain, self.pad_id_plain)
        cipher_tensor = self._pad_sequence(cipher_encoded, self.bos_id_cipher, self.eos_id_cipher, self.pad_id_cipher)
        
        return cipher_tensor, plain_tensor


class TokenFreeDataset(Dataset):
    """
    Dataset for Byte Latent Transformer (C5).
    Feeds raw bytes instead of tokens.
    """
    def __init__(self, plain_file, cipher_file, max_seq_len=2048, subset_size=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        plain_path = os.path.join(base_dir, 'Dataset_A1', plain_file)
        cipher_path = os.path.join(base_dir, 'Dataset_A1', cipher_file)
        
        self.max_seq_len = max_seq_len
        
        with open(plain_path, 'r', encoding='utf-8') as f:
            self.plain_lines = f.readlines()
            
        with open(cipher_path, 'r', encoding='utf-8') as f:
            self.cipher_lines = f.readlines()
            
        if subset_size is not None:
            self.plain_lines = self.plain_lines[:subset_size]
            self.cipher_lines = self.cipher_lines[:subset_size]
            
        assert len(self.plain_lines) == len(self.cipher_lines), "Mismatched line counts!"
        
        # 0: PAD, 1: BOS, 2: EOS, 3-258: Bytes
        self.pad_id_plain = 0
        self.bos_id_plain = 1
        self.eos_id_plain = 2
        
        self.plain_vocab_size = 259
        self.cipher_vocab_size = 259

    def __len__(self):
        return len(self.plain_lines)

    def _encode_bytes(self, text):
        # Convert text to UTF-8 bytes and shift by 3
        byte_array = text.encode('utf-8')
        return [b + 3 for b in byte_array]

    def _pad_sequence(self, token_ids, bos_id, eos_id, pad_id):
        token_ids = token_ids[:self.max_seq_len - 2]
        sequence = [bos_id] + token_ids + [eos_id]
        padding_length = self.max_seq_len - len(sequence)
        sequence = sequence + [pad_id] * padding_length
        return torch.tensor(sequence, dtype=torch.long)

    def __getitem__(self, idx):
        plain_line = self.plain_lines[idx].strip()
        cipher_line = self.cipher_lines[idx].strip()
        
        plain_encoded = self._encode_bytes(plain_line)
        cipher_encoded = self._encode_bytes(cipher_line)
        
        plain_tensor = self._pad_sequence(plain_encoded, self.bos_id_plain, self.eos_id_plain, self.pad_id_plain)
        cipher_tensor = self._pad_sequence(cipher_encoded, self.bos_id_plain, self.eos_id_plain, self.pad_id_plain)
        
        return cipher_tensor, plain_tensor
