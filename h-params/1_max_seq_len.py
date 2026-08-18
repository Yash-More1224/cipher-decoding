import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_and_measure(file_path, vocab_size=5000):
    """
    Trains a BPE tokenizer on the given file and returns the tokenizer
    along with the maximum sequence length (in tokens) found in the file.
    """
    print(f"Training tokenizer on {file_path}...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"], vocab_size=vocab_size)
    
    tokenizer.train([file_path], trainer)
    
    print(f"Tokenizing {file_path} to find max length...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    max_len = 0
    for line in lines:
        # Strip newlines as decided in our spec
        encoded = tokenizer.encode(line.strip())
        if len(encoded.ids) > max_len:
            max_len = len(encoded.ids)
            
    return tokenizer, max_len

if __name__ == "__main__":
    plain_file = '../Dataset_A1/brown_plain.txt'
    cipher_file = '../Dataset_A1/brown_cipher.txt'
    
    if not os.path.exists(plain_file):
        print(f"Error: Could not find {plain_file}. Please run this script from the h-params directory.")
        exit(1)

    plain_tok, plain_max = train_and_measure(plain_file)
    print(f"-> Max token length in plain text: {plain_max}\n")

    cipher_tok, cipher_max = train_and_measure(cipher_file)
    print(f"-> Max token length in cipher text: {cipher_max}\n")

    # +2 to accommodate the [BOS] and [EOS] tokens that will be appended
    max_seq_len = max(plain_max, cipher_max) + 2
    print(f"==================================================")
    print(f"RECOMMENDED max_seq_len (including BOS/EOS): {max_seq_len}")
    print(f"==================================================")

    # Save the trained tokenizers so we can reuse them later during model training
    os.makedirs('tokenizers', exist_ok=True)
    plain_tok.save("tokenizers/plain_tokenizer.json")
    cipher_tok.save("tokenizers/cipher_tokenizer.json")
    print("\nSaved tokenizers to h-params/tokenizers/")
