import sys
sys.path.append('..')

import transformers, datasets
from dataclasses import dataclass, field
from typing import List, Dict
import tensorflow as tf
from helper_scripts import argparser
# from ..helper_scripts.argparser import argparser

TASK_TO_FILTER_FN = {
    'qa': lambda example: example['task'] == 'qa',
    'qg': lambda example: example['task'] == 'qg',
    'e2e_qg': lambda example: example['task'] == 'e2e_qg',
    'ans_ext': lambda example: example['task'] == 'ans_ext',
    'multi': lambda example: example['task'] != 'e2e_qg' 
}



def encode_dataset_t5_qg(dataset, tokenizer, dataset_config): 
    def _add_special_tokens(example):
        example['source_text'] = example['source_text'].replace(
            "{hl_token}", '<hl>')    
        example['target_text'] = example['target_text'].replace(
            "{sep_token}", '<sep>')
        return example
    
    def _tokenize_function(example):
        #encoding
        source_encoding = tokenizer.encode_plus(
            example['source_text'],
            max_length=dataset_config["max_source_length"],
            padding = 'max_length',
            truncation=True
        )
        target_encoding = tokenizer.encode_plus(
            example['target_text'],
            max_length=dataset_config["max_source_length"],
            padding = 'max_length',
            truncation=True
        )

        input_ids = source_encoding['input_ids']
        input_attention = source_encoding['attention_mask']
        target_ids = target_encoding["input_ids"]
        target_attention = target_encoding["attention_mask"]

        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_attention,
            "labels":target_ids,
            "decoder_attention_mask": target_attention
        }
        return outputs

    dataset = dataset.map(_add_special_tokens)
    dataset = dataset.map(_tokenize_function)
    return dataset

def to_tf_dataset(dataset):
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    dataset.set_format('tf', columns = columns)
    return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
                'labels':tf.int32, 'decoder_attention_mask':tf.int32,  }
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                  'labels': tf.TensorShape([None]), 'decoder_attention_mask':tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
    return ds


def test():
    tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    tokenizer.add_special_tokens({'additional_special_tokens':['<hl>']})

    train_raw = datasets.load_dataset("../data/squad_multitask", name="highlight_qg_format", split='train[:10]')
    train_raw = encode_dataset_t5_qg(train_raw, tokenizer)

    it = iter(train_raw)
    # for i in range(10):
    #     # print("Example {}: {}".format(i, next(it)))
    #     print(l)
    #     print(next(it))
    print(len(train_raw))
    print(next(it))



dataset_processing_args = {
    "model_type":'t5',
    "model": "t5-small",
    "data_name":"highlight_qg_format",
    "task": 'qg',
    "qg_format": 'hl',
    "max_source_length": 512,
    "dataset_path": "../data/squad_multitask",
    "train_save_path": "../demos_data/train_data_qg_hl_t5_tf",
    "valid_save_path": "../demos_data/valid_data_qg_hl_t5_tf",
    "tokenizer_path":None
}

def main():
    args = argparser.get_dataset_arg_dict(dataset_processing_args)

    train_raw = datasets.load_dataset(
        args["dataset_path"], 
        name=args["data_name"], 
        split='train')
    valid_raw = datasets.load_dataset(
        args["dataset_path"], 
        name=args["data_name"], 
        split='validation')
    
    train_raw = train_raw.filter(TASK_TO_FILTER_FN[args["task"]])
    train_raw = valid_raw.filter(TASK_TO_FILTER_FN[args["task"]])

    tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    tokenizer.add_special_tokens({'additional_special_tokens':['<hl>']})

    train_raw = encode_dataset_t5_qg(train_raw, tokenizer, args)
    valid_raw = encode_dataset_t5_qg(valid_raw, tokenizer, args)

    train_tf = to_tf_dataset(train_raw)
    valid_tf = to_tf_dataset(valid_raw)

    tf.data.experimental.save(train_tf, args["train_save_path"])
    tf.data.experimental.save(valid_tf, args["valid_save_path"])


if __name__=="__main__":
    main()