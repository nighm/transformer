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
def translate(text, model, tokenizer, target_lang="zh", max_length=128):
    print(f"正在翻译文本: {text}")

    if target_lang == "zh":
        # 英文翻译为中文
        input_ids = tokenizer.encode(text, return_tensors='pt')
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=4,  # 可以使用的beam数量
            early_stopping=True  # 如果找到好的翻译就停止
        )
    elif target_lang == "en":
        # 中文翻译为英文
        input_ids = tokenizer.encode(text, return_tensors='pt', source_lang="zh")
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=4,  # 可以使用的beam数量
            early_stopping=True  # 如果找到好的翻译就停止
        )
    else:
        raise ValueError("目标语言应为 'zh' 或 'en'.")

    # 解码生成的文本
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"翻译完成。翻译结果: {translated_text}")
    return translated_text

# 选择翻译模式
while True:
    mode = input("请选择翻译模式：\n1. 中文翻译为英文\n2. 英文翻译为中文\n输入 'exit' 退出\n选择模式（1/2）：")

    if mode == "1":
        chinese_text = input("请输入要翻译的中文文本：")
        translation = translate(chinese_text, model, tokenizer, target_lang="en")
        print("\n原文:", chinese_text)
        print("翻译:", translation)
    elif mode == "2":
        english_text = input("请输入要翻译的英文文本：")
        translation = translate(english_text, model, tokenizer, target_lang="zh")
        print("\n原文:", english_text)
        print("翻译:", translation)
    elif mode.lower() == "exit":
        print("退出程序。")
        break
    else:
        print("无效的输入，请重新选择。")
