import tqdm
from comet_ml import Experiment
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp
import logging
from datasets import load_dataset
from grad_reversal_bert import RevGradBert
from transformers import BertTokenizerFast, BertForSequenceClassification, BertForTokenClassification
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import BertForSequenceClassification,BertTokenizerFast,AdamW,logging
logging.set_verbosity_error()
import torch
from livelossplot import PlotLosses
import imageio, glob

dim_reducer = TSNE(n_components=2)
#dim_reducer = PCA(n_components=2)

def visualize_layerwise_embeddings(hidden_states,masks,ys,epoch,title,layers_to_visualize=[0,1,2,3,8,9,10,11]):
    print('visualize_layerwise_embeddings for',title,'epoch',epoch)
    global dim_reducer
    num_layers = len(layers_to_visualize)
    fig = plt.figure(figsize=(24,int(num_layers/4)*6)) #each subplot of size 6x6
    ax = [fig.add_subplot(int(num_layers/4),4,i+1) for i in range(num_layers)]
    ys = ys.numpy().reshape(-1)
    for i,layer_i in enumerate(layers_to_visualize):#range(hidden_states):
        layer_hidden_states = hidden_states[layer_i]
        averaged_layer_hidden_states = torch.div(layer_hidden_states.sum(dim=1),masks.sum(dim=1,keepdim=True))
        layer_dim_reduced_vectors = dim_reducer.fit_transform(averaged_layer_hidden_states.numpy())
        df = pd.DataFrame.from_dict({'x':layer_dim_reduced_vectors[:,0],'y':layer_dim_reduced_vectors[:,1],'label':ys})
        df.label = df.label.astype(int)
        sns.scatterplot(data=df,x='x',y='y',hue='label', palette='Set1',ax=ax[i])
        fig.suptitle(f"{title}: epoch {epoch}")
        ax[i].set_title(f"layer {layer_i+1}")
    plt.savefig(f'plots/{title}.png',format='png',pad_inches=0)
    print()


model_name = "./models/singletask_model_language/lm"
# model_name = 'bert-base-multilingual-uncased'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# task_list = ["language"]
# label_map = {"language" : 7}
# dataset_dict = {
#     "language" : load_dataset('csv',data_files={'train' : '../datasets/common_crawl/language_identification_train.csv',
#     'test' : '../datasets/common_crawl/language_identification_test.csv'}),
# }

# language_to_int = {'en' : 0,'es' : 1,'nl' : 2,'pt' : 3,'zh' : 4,'tl' : 5,'hi' : 6}

# dataset_dict["language"] = dataset_dict["language"].map(lambda example: {"language": language_to_int[example['language']]})

max_seq_length = 128
batch_size = 16

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')

# def convert_to_language_features(example_batch):
#     inputs = list(example_batch['text'])
#     features = tokenizer.batch_encode_plus(
#         inputs, max_length=max_length, pad_to_max_length=True
#     )
#     features["labels"] = example_batch["language"]
#     return features

# convert_func_dict = {
#     "language": convert_to_language_features,
# }


# columns_dict = {
#     "language": ['input_ids', 'attention_mask', 'labels'],
# }

# features_dict = {}
# for task_name, dataset in dataset_dict.items():
#     features_dict[task_name] = {}
#     for phase, phase_dataset in dataset.items():
#         features_dict[task_name][phase] = phase_dataset.map(
#             convert_func_dict[task_name],
#             batched=True,
#             load_from_cache_file=False,
#         )
#         print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
#         features_dict[task_name][phase].set_format(
#             type="torch", 
#             columns=columns_dict[task_name],
#         )
#         print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

def get_bert_encoded_data_in_batches(df,batch_size = 16,max_seq_length = 128):
    global tokenizer
    data = [(row.text,row.label,) for _,row in df.iterrows()]
    sampler = torch.utils.data.sampler.SequentialSampler(data)
    batch_sampler = torch.utils.data.BatchSampler(sampler,batch_size=batch_size if batch_size > 0 else len(data), drop_last=False)
    for batch in batch_sampler:
        encoded_batch_data = tokenizer.batch_encode_plus([data[i][0] for i in batch],max_length = max_seq_length,pad_to_max_length=True,truncation=True)
        seq = torch.tensor(encoded_batch_data['input_ids'])
        mask = torch.tensor(encoded_batch_data['attention_mask'])
        yield (seq,mask),torch.LongTensor([data[i][1] for i in batch])

def load_df(path):
    ret_df = pd.read_csv(path,names=['text', 'language'])
    ret_df.drop(ret_df.index[0])
    ret_df['language'] = pd.Categorical(ret_df['language'], categories=['en','es','nl','pt','zh','tl','hi'])
    ret_df['label'] = ret_df['language'].cat.codes 
    print(ret_df)
    return ret_df.head(1500)

model = BertModel.from_pretrained(model_name)
model = model.to(device)
model.train(False)

val_df = load_df('../datasets/common_crawl/language_identification_test.csv')
epoch = 0

val_masks,val_ys = torch.zeros(0,max_seq_length),torch.zeros(0,1)
val_hidden_states = None

with torch.no_grad():
    for x,y in tqdm.tqdm(get_bert_encoded_data_in_batches(val_df,batch_size,max_seq_length)):
        sent_ids,masks = x
        sent_ids = sent_ids.to(device)
        masks = masks.to(device)
        y = y.to(device)
        model_out = model(sent_ids,masks,output_hidden_states=True,return_dict=True)
        hidden_states = model_out.hidden_states[1:]
        val_masks = torch.cat([val_masks,masks.cpu()])
        val_ys = torch.cat([val_ys,y.cpu().view(-1,1)])

        if type(val_hidden_states) == type(None):
            val_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
        else:
            val_hidden_states = tuple(torch.cat([layer_hidden_state_all,layer_hidden_state_batch.cpu()])for layer_hidden_state_all,layer_hidden_state_batch in zip(val_hidden_states,hidden_states))

    visualize_layerwise_embeddings(val_hidden_states,val_masks,val_ys,epoch,'val_data_revgradlanguage')
