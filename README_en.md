# Modern Chinese to Classical Chinese LLM

## Introduction

Input modern Chinese sentences and generate ancient Chinese style sentences. Based on the [Xunzi-Qwen2-1.5B base model](https://www.modelscope.cn/models/Xunzillm4cc/Xunzi-Qwen2-1.5B) and use some data from the "[Classical Chinese (Ancient Chinese) - Modern Chinese Parallel Corpus](https://github.com/NiuTrans/Classical-Modern)" to do LoRA fine-tuning. 

Welcome to try it out in the following links:
 - https://modelscope.cn/studios/chostem/ancient_Chinese_text_generator
 - https://huggingface.co/spaces/cofeg/ancient_Chinese_text_generator_1.5B

Model link:
 - ModelScope link: https://modelscope.cn/models/chostem/finetuned_Xunzi_Qwen2_1d5B_for_ancient_text_generation/
 - Hugging Face link: https://huggingface.co/cofeg/Finetuned-Xunzi-Qwen2-1.5B-for-ancient-text-generation/

## Complete LoRA fine-tuning workflow

1. Run `pip install -r requirements.txt` to install dependencies (pytorch is not listed, you need to check the CUDA version of your computer through the `nvcc --version` command first, and then go to the pytorch official website to install the corresponding CUDA version of pytorch)
2. Download the base model to be fine-tuned
3. Prepare modern-classical bitext data and put it in the data/original folder
4. Configure the base model path and name, data path, fine-tuned model storage path and name, whether to use cuda or cpu for training, etc. in the `config/config.py` file
5. Run `get_data.py` to process the original data into json format and divide it into training set, validation set, and test set
6. Set various training hyperparameters in the `finetune.py` file. Register a swanlab account in advance and get the api key to visualize the training process
7. Run `finetune.py` to start fine-tuning, and you will be prompted to enter the swanlab api key. During the fine-tuning process, the current checkpoint will be saved to the specified folder every `save_steps` step. If the training is interrupted for some reason, you can set `trainer.train(resume_from_checkpoint=True)` and then re-run `finetune.py` to continue training from the last checkpoint.
8. After fine-tuning is completed, you can select the best checkpoint based on the loss function on the swanlab website, or test the model on the validation set using the code in `evaluation.py`.
9. Use `merge_and_push_model.py` to merge the best checkpoint with the base model and export it as a complete fine-tuned model. If necessary, push it to hugging face (hugging face token required)
10. Use the code in `evaluation.py` to evaluate the effect of the final fine-tuned model on the test set, or run `web_demo_local_inference.py` in the demo folder to use the local model to generate text to check the effect. If the model has been pushed to hugging face, you can run `web_demo_api_inference.py` to use the hugging face's Serverless Inference API service for text generation.