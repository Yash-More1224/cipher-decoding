import os
from tokenizers import Tokenizer

def check_lengths():
    plain_tok_path = "outputs/plain_tokenizer.json"
    cipher_tok_path = "outputs/cipher_tokenizer.json"
    plain_file = "Dataset_A1/brown_plain.txt"
    cipher_file = "Dataset_A1/brown_cipher.txt"
    
    if not os.path.exists(plain_tok_path) or not os.path.exists(cipher_tok_path):
        print("Tokenizers not found in outputs/ directory.")
        return
        
    plain_tokenizer = Tokenizer.from_file(plain_tok_path)
    cipher_tokenizer = Tokenizer.from_file(cipher_tok_path)
    
    with open(plain_file, 'r', encoding='utf-8') as f:
        plain_lines = f.readlines()
        
    with open(cipher_file, 'r', encoding='utf-8') as f:
        cipher_lines = f.readlines()
        
    max_plain_len = 0
    max_cipher_len = 0
    
    def bits_to_chars(bit_string: str) -> str:
        chars = []
        bit_string = bit_string.strip()
        for i in range(0, len(bit_string), 8):
            chunk = bit_string[i:i+8]
            if len(chunk) == 8:
                chars.append(chr(int(chunk, 2) + 256))
        return "".join(chars)
    
    for line in plain_lines:
        line = line.strip()
        encoded = plain_tokenizer.encode(line).ids
        if len(encoded) > max_plain_len:
            max_plain_len = len(encoded)
            
    for line in cipher_lines:
        chars = bits_to_chars(line)
        encoded = cipher_tokenizer.encode(chars).ids
        if len(encoded) > max_cipher_len:
            max_cipher_len = len(encoded)
            
    print(f"Max Plaintext Token Length: {max_plain_len}")
    print(f"Max Ciphertext Token Length: {max_cipher_len}")

if __name__ == '__main__':
    check_lengths()
