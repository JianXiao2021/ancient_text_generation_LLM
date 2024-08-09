# 古文生成大模型

## 简介
基于[荀子基座大模型](https://github.com/Xunzi-LLM-of-Chinese-classics/XunziALLM)，采用“[文言文（古文）- 现代文平行语料](https://github.com/NiuTrans/Classical-Modern)”中的部分数据进行LoRA微调训练，可以将现代文句子转化为古文。

## 完整的LoRA微调工作流
1. 运行 `pip install -r requirements.txt` 安装依赖（pytorch未列出，需根据自己电脑硬件情况安装合适版本）
2. 下载要微调的基座大模型
3. 准备现代文-古文对照数据，放入 data/original 文件夹中
4. 在 `config/config.py` 文件中配置好基座模型路径和名称、数据路径、微调后的模型存放路径和名称、训练使用cuda还是cpu等
5. 运行 `get_data.py`，将原始数据处理为 json 格式并划分训练集、验证集、测试集
6. 在 `finetune.py` 文件中设置好各种训练超参。事先注册 swanlab 账号并拿到 api key，以便训练过程可视化
7. 运行 `finetune.py` 开始微调，会提示输入 swanlab api key。微调过程中每隔 `save_steps` 步会保存当前 checkpoint 到指定文件夹中。若训练因故中断，重新运行 `finetune.py` 会从最后一个 checkpoint 开始继续训练。
8. 微调完成，可根据 swanlab 网站上的 loss 图表，或自行修改 `evaluation.py` 中的代码在验证集上验证效果，从而挑选出最佳 checkpoint
9. 利用 `merge_and_push_model.py` 将最佳 checkpoint 与基座模型融合后导出为完整的微调模型，有需要的话推送到 hugging face 上（需有 hugging face token）
10. 利用 `evaluation.py` 中的代码在测试集上评估最终微调模型的效果，或运行 demo 文件夹中的 `web_demo_local_inference.py` 利用本地模型进行文本生成检查效果。如果模型已经推送到 hugging face，可以运行 `web_demo_api_inference.py` 利用 hugging face 的 Serverless Inference API 服务进行文本生成。