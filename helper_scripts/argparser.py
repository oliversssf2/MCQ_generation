import argparse
import math

default_model_args_dict = {
    "model_name_or_path":"t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "tokenizers/t5_qg_tokenizer",
    "train_file_path": "demos_data/train_data_e2e_qg_hl_t5",
    "valid_file_path": "demos_data/valid_data_e2e_qg_hl_t5"
}

default_training_args_dict = {
    "output_dir": "output_models/model_e2e_qg_hl_t5",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
    "seed": 42,
    "do_train": 1,
    "do_eval": 1,
    # "evaluate_during_training": 1,
    "logging_steps": 100,
    "prediction_loss_only": True,
    "fp16_full_eval": True
}

def get_arg_dicts(default_model_args_dict = default_model_args_dict, 
                 default_training_args_dict = default_training_args_dict):

    """Generate a dictionary of model argument and a dictionary of training arguments 

    Returns:
        [dict]: [two dictionaries]
    """    

    parser = argparse.ArgumentParser()

    for name, default_value in default_model_args_dict.items():
        parser.add_argument('--'+name, default=default_value, type=type(default_value))
        print(type(default_value))
    
    for name, default_value in default_training_args_dict.items():
        parser.add_argument('--'+name, default=default_value, type=type(default_value))
        # print(type(default_value))
    args = parser.parse_args()
    args_dict = vars(args)


    model_args_dict = {}
    training_args_dict = {}
    for name, value in args_dict.items():
        if name in default_model_args_dict:
            model_args_dict[name]=value
        elif name in default_training_args_dict:
            training_args_dict[name]=value

    return model_args_dict, training_args_dict

    # training_parser = argparse.ArgumentParser()

    # for name, default_value in default_training_args_dict.items():
    #     training_parser.add_argument('--'+name, default=default_value, type=type(default_value))
    #     print(type(default_value))
    # training_args = training_parser.parse_args()
    # return vars(model_args), vars(training_args)


if __name__ == '__main__':
    print(get_arg_dict())


