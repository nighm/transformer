import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 模型名称
model_name = "Helsinki-NLP/opus-mt-en-zh"

# 要保存模型的目录
save_directory = "D:/data/transformers/models/translation"

# 如果目录不存在，则创建它
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 打印模型和分词器是否加载成功
print("模型和分词器加载成功。")

# 定义翻译函数
def translate(text, model, tokenizer, max_length=128):
    print(f"正在翻译文本: {text}")

    # 编码输入文本
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # 生成翻译文本
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=4,  # 可以使用的beam数量
        early_stopping=True  # 如果找到好的翻译就停止
    )

    # 解码生成的文本
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"翻译完成。翻译结果: {translated_text}")
    return translated_text

# 手动输入要翻译的英文文本
english_text = input("请输入要翻译的英文文本: ")

# 执行翻译
chinese_translation = translate(english_text, model, tokenizer)

print("\n原文:", english_text)
print("翻译:", chinese_translation)
