from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp
import logging
from datasets import load_dataset
from grad_reversal_bert import RevGradBert


logging.basicConfig(level=logging.INFO)


model_name = "bert-base-multilingual-uncased"
task_list = ["language","emotion","pos"]
label_map = {"emotion" : 5, "language" : 7, "pos": 18}


dataset_dict = {
    "emotion" : load_dataset('csv',data_files={'train' : '../datasets/universal_joy/small.csv',
    'test' : '../datasets/universal_joy/test.csv'}),
    "language" : load_dataset('csv',data_files={'train' : '../datasets/common_crawl/language_identification_train.csv',
    'test' : '../datasets/common_crawl/language_identification_test.csv'}),
    "pos" : load_dataset('pos_dataset_loader.py',data_files={'train' : '../datasets/final_pos_data/train.conllu', 
    'test' : '../datasets/final_pos_data/test.conllu'})
}
emotion_to_int = {'anger' : 0,'anticipation' : 1 ,'fear' : 2,'joy' : 3,'sadness' : 4}
language_to_int = {'en' : 0,'es' : 1,'nl' : 2,'pt' : 3,'zh' : 4,'tl' : 5,'hi' : 6}

dataset_dict["emotion"] = dataset_dict["emotion"].map(lambda example: {"emotion": emotion_to_int[example['emotion']]})
dataset_dict["language"] = dataset_dict["language"].map(lambda example: {"language": language_to_int[example['language']]})

import transformers
from transformers import BertTokenizerFast, BertForSequenceClassification, BertForTokenClassification


max_length = 128
tokenizer = BertTokenizerFast.from_pretrained(model_name)

def convert_to_emotion_features(example_batch):
    inputs = list(example_batch['text'])
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["emotion"]
    return features

def convert_to_language_features(example_batch):
    inputs = list(example_batch['text'])
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["language"]
    return features

def convert_to_pos_features(example_batch):
    inputs = example_batch["tokens"]
    tokenized_inputs = tokenizer(inputs, max_length=max_length, pad_to_max_length=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(example_batch[f"upos"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:                            # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:              # Only label the first token of a given word.
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs    

convert_func_dict = {
    "language": convert_to_language_features,
    "emotion": convert_to_emotion_features,
    "pos": convert_to_pos_features
}


columns_dict = {
    "emotion": ['input_ids', 'attention_mask', 'labels'],
    "language": ['input_ids', 'attention_mask', 'labels'],
      "pos": ['input_ids', 'attention_mask', 'labels']
}


features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))


from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertTokenizer
from transformers import models
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BertModel,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)


metric = load_metric("accuracy")
seqeval_metric = load_metric("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


label_list = dataset_dict["pos"]["train"].features[f"upos"].feature.names
def compute_metrics_token(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#Use RevGradBert() instead of BertForSequenceClassification()
#For Gradient Reversal training for a particular task

####TRAINING LOOP 1 : LANGUAGE IDENTIFICATION####

# task = "pos"
# model = RevGradBertForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=label_map[task])
# training_args = TrainingArguments(output_dir="./models/singletask_model2_{}".format(task),
#     overwrite_output_dir=True,
#     learning_rate=1e-4,
#     do_train=True,
#     num_train_epochs=5,
#     per_device_train_batch_size=32,  
#     evaluation_strategy = 'epoch',
#     eval_steps = 500, # Evaluation and Save happens every X steps
#     save_total_limit = 3)
# trainer = Trainer(
# model=model, args=training_args, train_dataset=features_dict[task]["train"], eval_dataset=features_dict[task]["test"],compute_metrics=compute_metrics)
# trainer.train()

##SAVE THE UPDATED LM SEPERATELY###

# trained_lm = BertModel.from_pretrained("./models/singletask_model2_pos/checkpoint-1000")
# trained_lm.save_pretrained("./models/singletask_model2_pos/lm")

####TRAINING LOOP 2 : SECONDARY TASK####

task = "language"
# from transformers import DataCollatorForTokenClassification
# data_collator = DataCollatorForTokenClassification(tokenizer, padding=True, max_length=max_length)
model = BertForSequenceClassification.from_pretrained("./models/singletask_tokenmodel_pos/lm", num_labels=label_map[task])

#Freezing everything except classifier layer!
for name, param in model.named_parameters():
	if 'classifier' not in name: 
		param.requires_grad = False

training_args = TrainingArguments(output_dir="./models/singletask_model_{}".format(task),
    overwrite_output_dir=True,
    learning_rate=1e-4,
    do_train=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,  
    evaluation_strategy = 'epoch')
trainer = Trainer(
model=model, args=training_args, 
train_dataset=features_dict[task]["train"], eval_dataset=features_dict[task]["test"],compute_metrics=compute_metrics)
trainer.train()