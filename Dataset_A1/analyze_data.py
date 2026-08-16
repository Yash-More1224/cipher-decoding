# import sys

with open('brown_cipher.txt') as f:
    data = f.read().replace('\n', '').replace('\r', '')

with open('brown_plain.txt') as f:
    data_plain = f.read().replace('\n', '').replace('\r', '')

char_to_bits = {}
bits_to_chars = {}
i = 0

multi_bits_for_char = set()
multi_chars_for_bits = set()

for ch in data_plain:
    bits = data[i:i+8]
    
    # Check if one character maps to multiple different bit sequences
    if ch in char_to_bits:
        if char_to_bits[ch] != bits:
            multi_bits_for_char.add(ch)
    else:
        char_to_bits[ch] = bits
        
    # Check if one bit sequence maps to multiple different characters
    if bits in bits_to_chars:
        if bits_to_chars[bits] != ch:
            multi_chars_for_bits.add(bits)
    else:
        bits_to_chars[bits] = ch
        
    i += 8

print("Characters mapping to multiple bit sequences:")
print(sorted(list(multi_bits_for_char)))

print("\nNumber of bit sequences mapping to multiple characters:", len(multi_chars_for_bits))
# You can print them out if you want:
# print(sorted(list(multi_chars_for_bits)))

