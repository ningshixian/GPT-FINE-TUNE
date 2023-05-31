import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

# https://github.com/tloen/alpaca-lora/issues/333
model = LlamaForCausalLM.from_pretrained('./hf_ckpt', torch_dtype="auto", device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained('./hf_ckpt')

input_text = "What is the meaning of life?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
# tensor([[    1,  1724,   338,   278,  6593,   310,  2834, 29973]])

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text) #output: <s>What is the meaning of life? This is a question that has been asked by many people throughout

# ================================ #
# https://huggingface.co/shalomma/llama-7b-embeddings

# 获取 llama 编码的 sentence embedding
output = model.model.forward(input_ids)
print(output.last_hidden_state.shape)     # (1, 8, 4096)
print(output.last_hidden_state)     # [[ [],[],[] ]]
print(output.hidden_states)         # None
emb = torch.mean(output.last_hidden_state, dim=1)
print(emb.shape)    # (1, 4096)

# emb1 = model(**input_ids, output_hidden_states=True).last_hidden_state[:, 0]
# print(emb1)

# # 升级gcc到最新版本gcc-11.2.0
# # https://juejin.cn/post/7094518450386632711
# # pip install llama-cpp-python
# # 源码讲解：https://abetlen.github.io/llama-cpp-python/#python-bindings-for-llamacpp
# from llama_cpp import Llama
# llm = Llama(model_path="./hf_ckpt")
# output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
# print(output)
# embedding = llm.embed("Q: Name the planets in the solar system? A: ")
# print(embedding)