from datasets import Dataset
import os
from datasets import load_from_disk

class DatasetProcessor():
    def __init__(self, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    # 数据集基本情况和每行长度速览
    def dataset_quick_view(self, dataset: Dataset):
        print(dataset)
        sample = dataset.select(range(5))

        for row in sample:
            print(f"\n'>>> input: {row['input']}'")
            print(f"'>>> output: {row['output']}'")

        tokenized_input = self.tokenizer(sample["input"], add_special_tokens=False)
        tokenized_output = self.tokenizer(sample["output"], add_special_tokens=False)
        input_len = [len(row) for row in tokenized_input['input_ids']]
        output_len = [len(row) for row in tokenized_output['input_ids']]
        
        print(f"input length of sample: {input_len}")
        print(f"output length of sample: {output_len}")

    # The function for processing a single line of the dataset, it will be mapped to the whole dataset
    def process_func(self, sentence: dict) -> dict:

        input_ids, attention_mask, labels = [], [], []

        original_text = self.tokenizer(sentence['input'],
                                       add_special_tokens=False,
                                       max_length=self.max_source_length,
                                       truncation=True)
        translation = self.tokenizer(sentence['output'],
                                       add_special_tokens=False,
                                       max_length=self.max_target_length,
                                       truncation=True)

        # Concat input and output and a pad token (the same as eos token)
        input_ids = original_text["input_ids"] + translation["input_ids"] + [self.tokenizer.pad_token_id]
        # The model should learn when to generate the eos token and stop the output, so we set its attention mask to 1
        attention_mask = original_text["attention_mask"] + translation["attention_mask"] + [1]
        labels = [-100] * len(original_text["input_ids"]) + translation["input_ids"] + [self.tokenizer.pad_token_id]

        # Notice that the length of the three lists has not been padded to the max lenght yet.
        # It will be done later by the collator passed to the Trainer
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def get_tokenized_dataset(self, dataset: Dataset, base_path: str, name: str) -> Dataset:
        tokenized_dataset_path = os.path.join(base_path, f'tokenized_{name}_data')
        if os.path.exists(tokenized_dataset_path):
            tokenized_dataset = load_from_disk(tokenized_dataset_path)
            print(f"Attention: Load previously tokenized {name} dataset from disk. Eusure it matches the original dataset.")
        else:
            tokenized_dataset = dataset.map(
                self.process_func,
                # batched=True,  # sometimes batch processing will cause error
                remove_columns=dataset.column_names,
            )

            tokenized_dataset.save_to_disk(tokenized_dataset_path)
        return tokenized_dataset
