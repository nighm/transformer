from transformers import MarianMTModel, MarianTokenizer

# 加载模型和分词器
model_name = 'Helsinki-NLP/opus-mt-zh-en'  # 中文到英文翻译模型
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 翻译函数
def translate(text, model, tokenizer):
    # 将输入文本分词
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True)
    # 使用模型进行翻译
    translated = model.generate(inputs, max_length=512)
    # 解码翻译后的文本
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# 示例文本
text_zh = "你好，世界！"

# 翻译
text_en = translate(text_zh, model, tokenizer)
print(f"原文（中文）：{text_zh}")
print(f"译文（英文）：{text_en}")
