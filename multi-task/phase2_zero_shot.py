import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp
import logging
from datasets import load_dataset
from sklearn.metrics import f1_score



logging.basicConfig(level=logging.INFO)
model_name = "bert-base-multilingual-uncased"
task_list = ["emotion"]
label_map = {"emotion" : 5}
train_lang = 'en-es'
test_lang = 'es'

dataset_dict = {
    "emotion" : load_dataset('csv',data_files={'train' : '../datasets/universal_joy/train_{}.csv'.format(train_lang),
    'test' : '../datasets/universal_joy/test_{}.csv'.format(test_lang)})
}
emotion_to_int = {'anger' : 0,'anticipation' : 1 ,'fear' : 2,'joy' : 3,'sadness' : 4}

dataset_dict["emotion"] = dataset_dict["emotion"].map(lambda example: {"emotion": emotion_to_int[example['emotion']]})

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

convert_func_dict = {
    "emotion": convert_to_emotion_features
}


columns_dict = {
    "emotion": ['input_ids', 'attention_mask', 'labels']
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

# metric = load_metric("accuracy")
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
for task in task_list:
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=label_map[task])
    training_args = TrainingArguments(evaluation_strategy="epoch", output_dir="./models/singltask_model_{}".format(task),
        overwrite_output_dir=True,
        learning_rate=1e-5,
        do_train=True,
        num_train_epochs=20,
        per_device_train_batch_size=32,  
        save_steps=3000)
    trainer = Trainer(
    model=model, args=training_args, train_dataset=features_dict[task]["train"], eval_dataset=features_dict[task]["test"],compute_metrics=compute_metrics)
    trainer.train()
