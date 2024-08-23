import os
from huggingface_hub import InferenceClient
import gradio as gr
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import *

os.environ['CURL_CA_BUNDLE'] = ''  # Avoid SSL error
hugging_face_model_path = HUGGING_FACE_USER_NAME + "/" + MERGED_MODEL_NAME
client = InferenceClient(model=hugging_face_model_path, token=HUGGING_FACE_TOKEN)

def split_and_generate(modern_text):
    # Split the input text into sentences for the model is trained on sentence pairs
    sentences = re.findall(r'[^。！？]*[。！？]', modern_text)
    responses = ""
    for sentence in sentences:
        input = "现代文：" + sentence + " 古文："
        for token in client.text_generation(input, max_new_tokens=128, stream=True):
            if token != "<|endoftext|>":
                responses += token
                yield responses

demo = gr.Interface(fn=split_and_generate,
                    inputs=[gr.Textbox(label="现代文", lines=10)],
                    outputs=[gr.Textbox(label="古文", lines=10)],
                    title="现代文转古文大模型",
                    description="请在左边对话框输入你要转换的现代文并点击“Submit”按钮，耐心等待一两分钟，右边的对话框将显示转换后的古文。<br>由于训练数据来源于《徐霞客游记》，故对游记类文字转换效果较好。<br>详情请访问本项目[GitHub主页](https://github.com/JianXiao2021/ancient_text_generation_LLM)。"
)
demo.launch()
