git clone https://github.com/tloen/alpaca-lora.git

wget https://raw.githubusercontent.com/LC1332/Chinese-alpaca-lora/main/data/trans_chinese_alpaca_data.json

# LLaMA 模型下载
yum install git-lfs
# git lfs clone https://huggingface.co/decapoda-research/llama-7b-hf 
git lfs clone https://huggingface.co/yahma/llama-7b-hf

# 下载LLaMA的 lora权重：
git lfs clone https://huggingface.co/tloen/alpaca-lora-7b
# git lfs clone https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b
# git lfs clone https://huggingface.co/qychen/luotuo-lora-7b-0.1
