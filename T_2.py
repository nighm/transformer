import os
import time  # 导入time模块
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# 要保存模型的目录
save_directory = "D:/data/transformers/models/popular"

# 如果目录不存在，则创建它
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


# 定义下载和保存模型的函数
def download_and_save_model(model_name, save_path):
    try:
        print(f"开始下载模型: {model_name}")
        # 实际下载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 由于transformers库不提供下载进度的回调，我们无法显示实际的下载进度
        # 但我们可以显示一个简单的进度条，表示正在处理
        with tqdm(total=1, desc=f"Downloading {model_name}", unit='file') as pbar:
            save_path_model = os.path.join(save_path, model_name.replace("/", "_"))
            tokenizer.save_pretrained(save_path_model)
            model.save_pretrained(save_path_model)
            pbar.update(1)  # 更新进度条

        print(f"模型 {model_name} 下载并保存成功。")
    except Exception as e:
        print(f"下载模型 {model_name} 时发生错误: {e}")


# 流行的模型列表，去掉了 gpt2
popular_models = [
    "bert-base-uncased",
    "roberta-base",
    "t5-small",
    "facebook/bart-large-cnn",
    "google/electra-small-generator",
]

# 下载并保存所有流行模型
for model_name in popular_models:
    download_and_save_model(model_name, save_directory)