from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp
import logging
from datasets import load_dataset



logging.basicConfig(level=logging.INFO)


model_name = "bert-base-multilingual-uncased"
task_list = ["language"]
label_map = {"emotion" : 5, "language" : 7}


dataset_dict = {
    "emotion" : load_dataset('csv',data_files={'train' : '../datasets/universal_joy/small.csv',
    'test' : '../datasets/universal_joy/test.csv'}),
    "language" : load_dataset('csv',data_files={'train' : '../datasets/common_crawl/language_identification_train.csv',
    'test' : '../datasets/common_crawl/language_identification_test.csv'
})
}
emotion_to_int = {'anger' : 0,'anticipation' : 1 ,'fear' : 2,'joy' : 3,'sadness' : 4}
language_to_int = {'en' : 0,'es' : 1,'nl' : 2,'pt' : 3,'zh' : 4,'tl' : 5,'hi' : 6}

dataset_dict["emotion"] = dataset_dict["emotion"].map(lambda example: {"emotion": emotion_to_int[example['emotion']]})
dataset_dict["language"] = dataset_dict["language"].map(lambda example: {"language": language_to_int[example['language']]})

import transformers
from transformers import BertTokenizerFast, BertForSequenceClassification


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


convert_func_dict = {
    "language": convert_to_language_features,
    "emotion": convert_to_emotion_features
}


columns_dict = {
    "emotion": ['input_ids', 'attention_mask', 'labels'],
    "language": ['input_ids', 'attention_mask', 'labels']   
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



####TRAINING LOOP 1 : LANGUAGE IDENTIFICATION####

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=label_map["language"])
training_args = TrainingArguments(evaluation_strategy="epoch", output_dir="./models/singltask_model_{}".format("language"),
    overwrite_output_dir=True,
    learning_rate=1e-5,
    do_train=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,  
    save_steps=3000)
trainer = Trainer(
model=model, args=training_args, train_dataset=features_dict[task]["train"], eval_dataset=features_dict[task]["test"],compute_metrics=compute_metrics)
trainer.train()

####TRAINING LOOP 2 : EMOTION####

model = BertForSequenceClassification.from_pretrained("./models/singletask_model_language", num_labels=label_map["emotion"])
for name, param in model.named_parameters():
	if 'classifier' not in name: # classifier layer
		param.requires_grad = False
training_args = TrainingArguments(evaluation_strategy="epoch", output_dir="./models/singltask_model_{}".format("emotion"),
    overwrite_output_dir=True,
    learning_rate=1e-5,
    do_train=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,  
    save_steps=3000)
trainer = Trainer(
model=model, args=training_args, train_dataset=features_dict[task]["train"], eval_dataset=features_dict[task]["test"],compute_metrics=compute_metrics)
trainer.train()