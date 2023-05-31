import sys
import torch
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("../llama-7b-hf")
BASE_MODEL = "../llama-7b-hf"

tokenizer.pad_token_id = 0
print(tokenizer.eos_token_id)   # 2
print(tokenizer.bos_token_id)   # 1

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,     # V100建议不要开启
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
inputs = "你好,美国的首都在哪里？"
input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
print(input_ids)
generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=25,
        )
print(generation_output)
print(tokenizer.decode(generation_output[0]))

model = PeftModel.from_pretrained(
        model,
        "../lora-weights/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco",
        torch_dtype=torch.float16,
        device_map={'': 0}
    )

inputs = "你好,中国的首都在哪里？" #"你好,美国的首都在哪里？"
input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
print(input_ids)
generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=25,
        )
print(generation_output)
print(tokenizer.decode(generation_output[0]))
