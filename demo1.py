import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict

import datasets
from numpy.lib.utils import source
import transformers
from transformers import training_args
import torch
from transformers.models.clip.tokenization_clip import whitespace_clean
from transformers.utils.dummy_pt_objects import default_data_collator
from data_collator import T2TDataCollator

logger = logging.getLogger(__name__)


TASK_TO_FILTER_FN = {
    'qa': lambda example: example['task'] == 'qa',
    'qg': lambda example: example['task'] == 'qg',
    'e2e_qg': lambda example: example['task'] == 'e2e_qg',
    'ans_ext': lambda example: example['task'] == 'ans_ext',
    'multi': lambda example: example['task'] != 'e2e_qg' 
}

def load_hl_train_dataset_for_t5(dataset_config=None):
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

    # def _add_eos_examples(example):
    #     example['source_text'] = example['source_text'] + " </s>"
    #     example['target_text'] = example['target_text'] + " </s>"
    #     return example

    def _add_special_tokens(example):
        example['source_text'] = example['source_text'].replace(
            "{hl_token}", dataset_config.additional_special_tokens[0])    
        example['target_text'] = example['target_text'].replace(
            "{sep_token}", dataset_config.spectial_tokens['sep_token'])
        return example

    def _tokenize_function(example_batch):
        source_encoding = tokenizer.encode_plus(
            example_batch['source_text'],
            max_length=dataset_config.max_source_length,
            padding = 'max_length',
            truncation=True
        )
        target_encoding = tokenizer.encode_plus(
            example_batch['target_text'],
            max_length=dataset_config.max_source_length,
            padding = 'max_length',
            truncation=True
        )
        encodings = {
            'source_ids': source_encoding['input_ids'],
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask']
        }

        return encodings

    if dataset_config == None:
        dataset_config = DataTrainingArguments()

    train_raw = datasets.load_dataset('data/squad_multitask/', name='highlight_qg_format', split='train')
    valid_raw = datasets.load_dataset('data/squad_multitask/', name='highlight_qg_format', split='validation')
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
    # tokenizer.add_special_tokens(training_config.spectial_tokens)
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    tokenizer.add_special_tokens({'additional_special_tokens':['<hl>']})

    #Select the entries for the intended task
    train_raw = train_raw.filter(TASK_TO_FILTER_FN[dataset_config.task])
    valid_raw = valid_raw.filter(TASK_TO_FILTER_FN[dataset_config.task])

    #Preprocessing and tokenization
    # train_raw = train_raw.map(_add_eos_examples)
    train_raw = train_raw.map(_add_special_tokens)
    train_raw = train_raw.map(_tokenize_function, batched=False)
    
    # valid_raw = valid_raw.map(_add_eos_examples)
    valid_raw = valid_raw.map(_add_special_tokens)
    valid_raw = valid_raw.map(_tokenize_function, batched=False)

    return train_raw, valid_raw, tokenizer, dataset_config

def save_train_valid_tok(train_data, valid_data, tokenizer, training_config):
    if training_config.train_file_name is None:
        train_file_name = f'train_data_{training_config.task}_{training_config.qg_format}_{training_config.model_type}'
    else: train_file_name = training_config.train_file_name

    if training_config.valid_file_name is None:
        valid_file_name = f'valid_data_{training_config.task}_{training_config.qg_format}_{training_config.model_type}'
    else: valid_file_name = training_config.valid_file_name

    if training_config.tokenizer_file_name is None:
        tokenizer_file_name = f'tokenizer_{training_config.qg_format}_{training_config.model_type}'
    else: tokenizer_file_name = training_config.tokenizer_file_name

    if training_config.path is None:
        path = f'data_{training_config.task}_{training_config.qg_format}_{training_config.model_type}'
    else: path = training_config.path

    train_data.save_to_disk(os.path.join(path, train_file_name))
    logger.info('saved training set at {}'.format(os.path.join(path, train_file_name)))

    valid_data.save_to_disk(os.path.join(path, valid_file_name))
    logger.info('saved validation set at {}'.format(os.path.join(path, valid_file_name)))

    tokenizer.save_pretrained(os.path.join(path, tokenizer_file_name))
    logger.info('saved tokenizer at {}'.format(os.path.join(path, tokenizer_file_name)))
    
def train():
    args_dict = {
        "model_name_or_path": "t5-small",
        "model_type": "t5",
        "tokenizer_name_or_path": "demos_data/tokenizer_hl_t5",
        "output_dir": "output_data/model_e2e_qg_hl_t5",
        "train_file_path": "demos_data/train_data_e2e_qg_hl_t5",
        "valid_file_path": "demos_data/valid_data_e2e_qg_hl_t5",
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 10,
        "seed": 42,
        "do_train": True,
        "do_eval": True,
        "evaluate_during_training": True,
        "logging_steps": 100
    }

    training_args = transformers.TrainingArguments(
        output_dir=args_dict['output_dir'],
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=3,
        learning_rate=1e-4,
        gradient_accumulation_steps=8,
        do_train=True,
        do_eval=True,
        # evaluate_during_training=True,
        logging_steps=100,
        evaluation_strategy='epoch',
        prediction_loss_only=True,
        fp16_full_eval=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args_dict['tokenizer_name_or_path'])
    
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args_dict['model_name_or_path']
    )
    model.resize_token_embeddings(len(tokenizer))

    train_set = datasets.load_from_disk(args_dict['train_file_path'])
    train_set = train_set.remove_columns(['source_text', 'target_text','task'])
    train_set = train_set.rename_column('source_ids', 'input_ids')
    train_set = train_set.rename_column('target_ids', 'labels')
    
    train_set.set_format('torch')

    valid_set = datasets.load_from_disk(args_dict['valid_file_path'])
    valid_set = valid_set.remove_columns(['source_text', 'target_text', 'task'])
    valid_set = valid_set.rename_column('source_ids', 'input_ids')
    valid_set = valid_set.rename_column('target_ids', 'labels')
    valid_set.set_format('torch')

    def compute_metrics(eval_preds):
        metrics = datasets.load_metrics('bleu', 'meteor', 'rouge', 'f1')

    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=args_dict['model_type'],
        mode="training",
        # using_tpu=False
    )

    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        train_dataset = train_set,
        eval_dataset = valid_set,
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()

def print_dataset_summary(ds, name):
    print('The features of {} are: {}'.format(name, ds.features))
    print('The length of {} is: {}'.format(name, len(ds)))

def test_generated_dataset(ds_path, tokenizer_path, name=None):
    data = datasets.load_from_disk(ds_path)
    print(data.features)
    print(len(data))
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    import numpy as np
    for i in range(10):
        id = np.random.randint(0, len(data))
        print(tokenizer.decode(data[id]['source_ids']))
        print(tokenizer.decode(data[id]['target_ids']))

def create_and_save_dataset():
    train_data, valid_data, tokenizer, training_config =\
        load_hl_train_dataset_for_t5()
    print_dataset_summary(train_data, 'training_set')
    print_dataset_summary(valid_data, 'valid_set')
    save_train_valid_tok(train_data, valid_data, tokenizer, training_config)

def simple_example():
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained('output_data/model_qg_hl_t5')
    tokenizer = transformers.AutoTokenizer.from_pretrained('t5_qg_tokenizer')

    eval = datasets.load_from_disk('demos_data/valid_data_qg_hl_t5/')
    eval_clean = eval.remove_columns(['source_text', 'target_text','task', 'target_ids'])
    eval_clean = eval_clean.rename_column('source_ids', 'input_ids')
    eval_clean.set_format('torch')

    print(eval_clean[0])


    outs = model.generate(
        input_ids=[eval_clean[0:3]['input_ids']], 
        attention_mask=eval_clean[0:3]['attention_mask'], 
        max_length=32,
        num_beams=4,
    )
    # print(tokenizer.decode(eval_clean[0:3]['input_ids']))
    # for i in range(3):
    print(tokenizer.batch_decode(outs, skip_special_tokens=True))
    # print(outs)

if __name__ == "__main__":
    # create_and_save_dataset()
    # main()
    # test_generated_dataset('demos_data/train_data_ans_ext_hl_t5/', 'demos_data/tokenizer_hl_t5/')
    train()
    # simple_example()

    # eval = datasets.load_from_disk('demos_data/valid_data_qg_hl_t5/')
    # print(eval.features)


