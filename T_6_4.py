import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 模型名称
model_name = "Helsinki-NLP/opus-mt-en-zh"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 定义翻译函数
def translate(text, target_lang="zh", max_length=128):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Streamlit界面
st.title("翻译器")

st.sidebar.header("翻译设置")
source_lang = st.sidebar.radio("选择源语言", ("English", "Chinese"))

# 用户输入文本
source_text = st.text_input("输入文本：", "")

# 翻译按钮
if st.button("翻译"):
    if not source_text:
        st.warning("请输入要翻译的文本。")
    else:
        try:
            target_lang = "en" if source_lang == "English" else "zh"
            translation = translate(source_text, target_lang=target_lang)
            st.success(f"翻译结果: {translation}")
        except Exception as e:
            st.error(f"b翻译出错: {e}")