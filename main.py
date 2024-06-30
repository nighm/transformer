import os
from transformers import AutoModel, AutoTokenizer

save_directory = "D:/data/transformers/models/facebook"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)


def download_and_save_model(model_name, save_path):
    try:
        print(f"开始下载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        save_path_model = os.path.join(save_path, model_name.replace("/", "_"))
        tokenizer.save_pretrained(save_path_model)
        model.save_pretrained(save_path_model)
        print(f"模型 {model_name} 下载并保存成功到 {save_path_model}。")
    except Exception as e:
        print(f"下载模型 {model_name} 时发生错误: {e}")


facebook_models = [
    "facebook/bart-large-cnn",
    # 添加更多Facebook模型名称
]

for model_name in facebook_models:
    download_and_save_model(model_name, save_directory)