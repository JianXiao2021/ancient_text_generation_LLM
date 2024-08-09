import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.generate import generate_answer
from config.config import *

fine_tuned_model_path = os.path.join(FINE_TUNED_MODELS_PATH, MERGED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, torch_dtype="auto", device_map=DEVICE)
model.generation_config.pad_token_id = tokenizer.pad_token_id  # To avoid warnings

def split_and_generate(modern_text, progress=gr.Progress()):
    progress(0, desc="开始处理")
    # Split the input text into sentences for the model is trained on sentence pairs
    sentences = re.findall(r'[^。！？]*[。！？]', modern_text)
    responses = ""
    for sentence in progress.tqdm(sentences, desc="生成中……"):
        input = "现代文：" + sentence + " 古文："
        response = generate_answer(input, tokenizer, DEVICE, model)
        responses += response
    return responses

demo = gr.Interface(fn=split_and_generate,
                        inputs=[gr.Textbox(label="现代文", lines=10)],
                        outputs=[gr.Textbox(label="古文", lines=10)])
demo.launch()
