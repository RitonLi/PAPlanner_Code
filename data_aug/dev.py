

import json
import os
import os.path as osp
import shutil
import sys

import torch
import torch.distributed as dist
from filelock import FileLock, Timeout
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_id = "Qwen/Qwen3-VL-7B"
local_rank = 2

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "load_in_4bit": False,
        "device_map": f"cuda:{local_rank}",
    },  # "device_map": "auto"},
    return_full_text=False,
    repetition_penalty=1.0,
)

generation_config = {
    "temperature": 0.2,
    "top_p": 0.6,
    "do_sample": True,
    "max_new_tokens": 256,
}

print(model_id)
print(generation_config)

while True:
    print("--" * 50)
    # input_msg = input("Please enter inputs:\n")
    input_msg = """Please reverse the order of words in the sentence.

    result = pipe(input_msg + "\n", **generation_config)
    print("--" * 50)
    print(result[0]["generated_text"])
    break
