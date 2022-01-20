from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp
import logging
from datasets import load_dataset



logging.basicConfig(level=logging.INFO)

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

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
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


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        ## add task specific output heads
        self.classifier1 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[0]
        )
        self.classifier2 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[1]
        )

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = None
        if task_name == list(self.num_labels.keys())[0]:
            logits = self.classifier1(pooled_output)
        elif task_name == list(self.num_labels.keys())[1]:
            logits = self.classifier2(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels[task_name]), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model_name = "bert-base-multilingual-uncased"
multitask_model = BertForSequenceClassification.from_pretrained(
        model_name,
        task_labels_map={"emotion": 5, "language": 7},
    )
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


max_length = 128

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


import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
import numpy as np
import torch
import transformers


class NLPDataCollator:
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """

    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.float
                    )
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):
    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    # def get_eval_dataloader(self):
    #     return MultitaskDataloader(
    #         {
    #             task_name: self.get_single_train_dataloader(task_name, task_dataset)
    #             for task_name, task_dataset in self.train_dataset.items()
    #         }
    #     )


train_dataset = {
    task_name: dataset["train"] 
    for task_name, dataset in features_dict.items()
}
eval_dataset = {
    task_name: dataset["test"]
    for task_name, dataset in features_dict.items()
}
trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        output_dir="./models/multitask_model",
        overwrite_output_dir=True,
        learning_rate=1e-5,
        do_train=True,
        num_train_epochs=20,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=32,  
        save_steps=3000
    ),
    data_collator=NLPDataCollator(),
    train_dataset=train_dataset
)
trainer.train()


preds_dict = {}
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
for task_name in ["emotion", "language"]:
    val_len = len(features_dict[task_name]["test"])
    acc = 0.0
    buggy = 0
    for index in range(0, val_len):

        try:
            labels = features_dict[task_name]["test"][index]["labels"]
        except:
            continue
            buggy+=1
        inputs = features_dict[task_name]["test"][index]["input_ids"]

        logits = multitask_model(inputs.unsqueeze(0).cuda(), task_name=task_name)[0]

        predictions = torch.argmax(
            torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),
            dim=1,
        )
        acc += sum(np.array(predictions) == np.array(labels))
    acc = acc / val_len
    print(f"Task name: {task_name} \t Accuracy: {acc}")
    print("Buggy Samples: {}".format(buggy))