# Reference: 
# https://discuss.huggingface.co/t/further-finetuning-a-lora-finetuned-causallm-model/3698 7/4
# https://github.com/TrelisResearch/install-guides/blob/main/Pushing_to_Hub.ipynb

from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login
from config.config import *

best_checkpoint_path = os.path.join(FINE_TUNED_MODELS_PATH, BASE_MODEL_NAME + "_checkpoints", "checkpoint-5400")
merged_model_path = os.path.join(FINE_TUNED_MODELS_PATH, MERGED_MODEL_NAME)

if os.path.exists(merged_model_path):
    model_to_push = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype="auto", device_map=DEVICE)
else:
    base_with_adapters_model = AutoPeftModelForCausalLM.from_pretrained(best_checkpoint_path, torch_dtype="auto", device_map=DEVICE)
    ## Or use the following code to load the base model and the adapter separately:
    # base_model_path = base_model_path = os.path.join(BASE_MODELS_PATH, BASE_MODEL_NAME)
    # base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map=DEVICE)
    # model_to_push = PeftModel.from_pretrained(base_model,best_checkpoint_path) # Apply the desired adapter to the base model

    model_to_push = base_with_adapters_model.merge_and_unload()  # merge adapters with the base model
    model_to_push.save_pretrained(merged_model_path)  # Save the merged model locally
    base_model_path = os.path.join(BASE_MODELS_PATH, BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)  # Save the tokenizer from the base model, it's necessary for the model to work


login(HUGGING_FACE_TOKEN)
model_to_push.push_to_hub(MERGED_MODEL_NAME, token=True, max_shard_size="5GB", safe_serialization=True)