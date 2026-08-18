import torch
import os

print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Is CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
