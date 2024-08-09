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
                        outputs=[gr.Textbox(label="古文", lines=10)])
demo.launch()
