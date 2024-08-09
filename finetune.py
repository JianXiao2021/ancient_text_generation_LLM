from datasets import load_dataset
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from utils.data_processor import DatasetProcessor
from config.config import *

# 训练参数设置
max_source_length = 256  # max length of the original text
max_target_length = 128  # max length of the target text
target_modules = ["q_proj", "k_proj", "v_proj"]
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.1
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
num_train_epochs = 3
logging_steps = 50
save_steps = 600
learning_rate = 1e-4
do_eval = False  # 暂时没搞明白compute_metrics怎么写，没法在训练过程中评估
output_dir = os.path.join(FINE_TUNED_MODELS_PATH, BASE_MODEL_NAME + '_checkpoints')
swanlab_project_name = BASE_MODEL_NAME.replace(".", "_") + "_finetune_project"
swanlab_experiment_name = BASE_MODEL_NAME.replace(".", "_") + "_finetune_experiment"

# 加载预训练模型和分词器
model_path = os.path.join(BASE_MODELS_PATH, BASE_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=DEVICE, torch_dtype="auto")
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

# 加载和查看数据集
data_path = os.path.join(DATA_PATH, 'processed')
train_dataset = load_dataset("json", data_files = os.path.join(data_path, 'train_data.json'), split='train')
test_dataset = load_dataset("json", data_files = os.path.join(data_path, 'test_data.json'), split='train')  # The default name of dataset split is 'train'
data_prcocessor = DatasetProcessor(tokenizer, max_source_length, max_target_length)
data_prcocessor.dataset_quick_view(train_dataset)
data_prcocessor.dataset_quick_view(test_dataset)

# 得到微调数据集
tokenized_train_dataset = data_prcocessor.get_tokenized_dataset(train_dataset, data_path, 'train')

tokenized_validation_dataset = None
if do_eval:
    validation_dataset_path = os.path.join(data_path, 'validation_data.json')
    if os.path.exists(validation_dataset_path):
        validation_dataset = load_dataset("json", data_files = validation_dataset_path, split='train')
        tokenized_validation_dataset = data_prcocessor.get_tokenized_dataset(validation_dataset, data_path, 'validation')

# 创建LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
    inference_mode=False,  # 训练模式
    r=lora_rank,  # Lora 秩
    lora_alpha=lora_alpha,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=lora_dropout,  # Dropout 比例
)

# 将LoRA应用于模型
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 创建微调参数
args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=logging_steps,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    learning_rate=learning_rate,
    gradient_checkpointing=True,
)

# SwanLab微调过程回调数据
swanlab_callback = SwanLabCallback(project=swanlab_project_name, experiment_name=swanlab_experiment_name)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # compute_metrics=compute_metrics,  # 暂时没搞明白compute_metrics怎么写，所以训练时实际上并没有在验证集上评估
    callbacks=[swanlab_callback],
)

# 开始微调
trainer.train(resume_from_checkpoint = True)
print("微调完成！")

swanlab.finish()
