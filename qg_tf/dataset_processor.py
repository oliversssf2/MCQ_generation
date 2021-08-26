from datasets import load_dataset
import tensorflow_datasets as tfds
import datasets
from transformers import AutoTokenizer
import os

from ..helper_scripts import argparser



data_dir = "data/squad_multitask/"
data_name="highlight_qg_format"

TASK_TO_FILTER_FN = {
    'qa': lambda example: example['task'] == 'qa',
    'qg': lambda example: example['task'] == 'qg',
    'e2e_qg': lambda example: example['task'] == 'e2e_qg',
    'ans_ext': lambda example: example['task'] == 'ans_ext',
    'multi': lambda example: example['task'] != 'e2e_qg' 
}

@dataclass
class DataTrainingArguments:
    model_type='t5'
    task: str = 'e2e_qg'
    qg_format: str = 'hl'
    max_source_length: int = 512
    spectial_tokens: dict = field(default_factory=\
        lambda:{'sep_token': '<sep>'})

    additional_special_tokens: List = field(default_factory = lambda:['<hl>'])
    train_file_name: str = None
    valid_file_name: str = None
    tokenizer_file_name: str = None
    path: str = 'demos_data'


def main():
    dataset_config = DataTrainingArguments()

    train_raw = datasets.load_dataset(data_dir, name=data_name, split='train')
    valid_raw = datasets.load_dataset(data_dir, name=data_name, split='validation')
    train_raw.filter(TASK_TO_FILTER_FN[dataset_config.task])
    valid_raw.filter(TASK_TO_FILTER_FN[dataset_config.task])

    tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    tokenizer.add_special_tokens({'additional_special_tokens':['<hl>']})


if __name__ == "__main__":
    main()