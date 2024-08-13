from config.config import BASE_MODEL_NAME

# Training Qwen2-7B on RTX 4090
if BASE_MODEL_NAME == "Xunzi-Qwen2-7B":
    lora_config_dict = {
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "inference_mode": False,  # 训练模式
        "r": 8,  # Lora 秩
        "lora_alpha": 16,
        "lora_dropout": 0.1
    }
    traning_args_dict = {
        "eval_strategy": "no",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 5,
        "logging_steps": 50,
        "save_steps": 600,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "do_eval": False,  # 暂时没搞明白compute_metrics怎么写，没法在训练过程中评估
        "gradient_checkpointing": True,
    }
# Training Qwen2-1.5B on RTX 2060
elif BASE_MODEL_NAME == "Xunzi-Qwen2-1.5B":
    lora_config_dict = {
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "inference_mode": False,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1
    }
    traning_args_dict = {
        "eval_strategy": "no",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "logging_steps": 50,
        "save_steps": 600,
        "learning_rate": 1e-4,
        "do_eval": False,
        "gradient_checkpointing": True,
    }