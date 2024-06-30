import os
import tkinter as tk
from tkinter import ttk, messagebox
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

# 定义翻译函数
def translate(text, target_lang="zh", max_length=128):
    print(f"正在翻译文本: {text}")
    input_ids = tokenizer.encode(text, return_tensors='pt')
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"翻译完成。翻译结果: {translated_text}")
    return translated_text

# 创建主窗口
root = tk.Tk()
root.title("翻译器")

# 创建变量存储用户输入和翻译结果
source_text = tk.StringVar()
target_text = tk.StringVar()

# 创建源语言和目标语言的标签和文本框
ttk.Label(root, text="输入文本：").grid(row=0, column=0)
source_entry = ttk.Entry(root, textvariable=source_text, width=50)
source_entry.grid(row=0, column=1)

ttk.Label(root, text="翻译结果：").grid(row=1, column=0)
target_entry = ttk.Entry(root, textvariable=target_text, width=50, state='readonly')
target_entry.grid(row=1, column=1)

# 创建按钮进行翻译
def translate_button_clicked():
    try:
        translation = translate(source_text.get(), target_lang="en" if source_lang.get() == "English" else "zh")
        target_text.set(translation)
    except Exception as e:
        messagebox.showerror("错误", str(e))

source_lang = tk.StringVar(value="English")
ttk.Radiobutton(root, text="English to Chinese", variable=source_lang, value="English").grid(row=2, column=0, sticky='w')
ttk.Radiobutton(root, text="Chinese to English", variable=source_lang, value="Chinese").grid(row=2, column=1, sticky='w')

translate_button = ttk.Button(root, text="翻译", command=translate_button_clicked)
translate_button.grid(row=3, column=0, columnspan=2)

# 运行主循环
root.mainloop()