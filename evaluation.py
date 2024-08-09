import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from rouge_chinese import Rouge
import jieba
import csv
from config.config import *
from utils.generate import generate_answer

class ModelEvaluator:
    def __init__(self, dataset: Dataset, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.references = self.dataset['output']
        self.predictions = []

    def get_predictions(self, model):
        model.generation_config.pad_token_id = self.tokenizer.pad_token_id  # To avoid warnings
        predictions = []
        for item in tqdm(self.dataset):
            prediction = generate_answer(item['input'], self.tokenizer, DEVICE, model)
            predictions.append(prediction)
        model.to('cpu')
        torch.cuda.empty_cache()
        return predictions

    def get_rogue_evaluation(self) -> list:
        print(f"\nModel {os.path.basename(self.model.name_or_path)} is predicting...\n")
        self.predictions = self.get_predictions(self.model)
        num_sentences = len(self.references)
        assert len(self.predictions) == num_sentences

        avg_metrics = {
            'rouge-1': {'r': 0, 'p': 0, 'f': 0},
            'rouge-2': {'r': 0, 'p': 0, 'f': 0},
            'rouge-l': {'r': 0, 'p': 0, 'f': 0}
        }
        result = []

        # Calculate the ROUGE scores for each sentence
        for i in range(num_sentences):
            hypothesis = ' '.join(jieba.cut(self.predictions[i]))
            reference = ' '.join(jieba.cut(self.references[i]))
            rouge = Rouge()
            scores = rouge.get_scores(hypothesis, reference)
            scores = scores[0]
            for key in avg_metrics:
                for sub_key in avg_metrics[key]:
                    avg_metrics[key][sub_key] += scores[key][sub_key]
            # save scores of a single sentence to the result list
            result.append(scores)
        
        # Calculate the average score and append it to the result list
        for key in avg_metrics:
            for sub_key in avg_metrics[key]:
                avg_metrics[key][sub_key] /= num_sentences
        result.append(avg_metrics)

        print("平均指标:")
        for key in avg_metrics:
            print(f"{key}: Recall: {avg_metrics[key]['r']:.4f}, Precision: {avg_metrics[key]['p']:.4f}, F1 Score: {avg_metrics[key]['f']:.4f}")
        return result

    def save_rouge_result(self, results: list, filename: str):
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['Prediction', 'Reference', 'ROUGE-1 RECALL', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1'])
            # 写入每个句子的指标
            for i in range(len(results) - 1):
                writer.writerow([
                    self.predictions[i],
                    self.references[i],
                    results[i]['rouge-1']['r'],
                    results[i]['rouge-1']['f'],
                    results[i]['rouge-2']['f'],
                    results[i]['rouge-l']['f']
                ])
            # 写入平均指标
            avg_metrics = results[-1]
            writer.writerow(['平均', '', avg_metrics['rouge-1']['r'], avg_metrics['rouge-1']['f'], avg_metrics['rouge-2']['f'], avg_metrics['rouge-l']['f']])


def EvaluateCausalLM(model_path: str, is_checkpoint: bool, tokenizer_path: str, dataset: Dataset, output_file_name: str):

    if is_checkpoint:
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_evaluator = ModelEvaluator(dataset, tokenizer, model)

    base_model_result = model_evaluator.get_rogue_evaluation()
    model_evaluator.save_rouge_result(base_model_result, output_file_name)


if __name__ == '__main__':
    # Load test dataset
    data_path = os.path.join(DATA_PATH, 'processed')
    test_dataset = load_dataset("json", data_files = os.path.join(data_path, 'test_data.json'), split='train')  # The default name of dataset split is 'train'
    test_dataset = Dataset.from_dict(test_dataset[:3])

    # Evaluate the best checkpoint, using the tokenizer of the base model
    check_point_path = os.path.join(FINE_TUNED_MODELS_PATH, BASE_MODEL_NAME + "_checkpoints", "checkpoint-5400")
    tokenizer_path = os.path.join(BASE_MODELS_PATH, BASE_MODEL_NAME)
    EvaluateCausalLM(check_point_path, True, tokenizer_path, test_dataset, 'checkpoint_5400_evaluation.csv')

    # Evaluate base model
    base_model_path = os.path.join(BASE_MODELS_PATH, BASE_MODEL_NAME)
    EvaluateCausalLM(base_model_path, False, base_model_path, test_dataset, 'base_model_evaluation.csv')

    # # Evaluate merged fine-tuned model
    # fine_tuned_model_path = os.path.join(FINE_TUNED_MODELS_PATH, MERGED_MODEL_NAME)
    # EvaluateCausalLM(fine_tuned_model_path, False, fine_tuned_model_path, test_dataset, 'finetuned_model_evaluation.csv')