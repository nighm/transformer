from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 选择适合翻译任务的预训练模型，这里使用mbart模型作为示例
model_name = "facebook/mbart-large-50-many-to-many-mmt"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 定义翻译函数
def translate(text, model, tokenizer):
    # 编码输入文本，指定源语言和目标语言
    source_text = f"translate English to Chinese: {text}"
    input_ids = tokenizer.encode(source_text, return_tensors='pt')

    # 生成翻译文本，使用max_new_tokens控制生成的最大长度
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50)

    # 解码生成的文本
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# 要翻译的英文文本
english_text = "Hello, world!"

# 执行翻译
chinese_translation = translate(english_text, model, tokenizer)

print("原文:", english_text)
print("翻译:", chinese_translation)