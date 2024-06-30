import os
from transformers import AutoModel, AutoTokenizer

# 要保存模型的目录
save_directory = "D:/data/transformers/models/small"

# 如果目录不存在，则创建它
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


# 定义下载和保存模型的函数
def download_and_save_model(model_name, save_path):
    try:
        print(f"开始下载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 保存分词器和模型到指定路径
        save_path_model = os.path.join(save_path, model_name)
        tokenizer.save_pretrained(save_path_model)
        model.save_pretrained(save_path_model)
        print(f"模型 {model_name} 下载并保存成功。")
    except Exception as e:
        print(f"下载模型 {model_name} 时发生错误: {e}")


# 较小的模型列表
small_models = [
    "distilbert-base-uncased",  # DistilBERT
    # "huawei-noah/TinyBERT_General_6L_768H",  # TinyBERT
    # "albert-base-v1",  # ALBERT
    # "google/mobilebert-uncased",  # MobileBERT
    # "google/pegasus-xsum",  # BERT Pegasus
]

# 下载并保存所有小模型
for model_name in small_models:
    download_and_save_model(model_name, save_directory)