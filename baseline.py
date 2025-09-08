# -*- coding: utf-8 -*-
"""
Script for running the LongBERT baselines with optimization.

Copyright (c) 2025 Idiap Research Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import json
import torch
import optuna
import datasets
import argparse
import numpy as np

from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from main import load_dataset

parser = argparse.ArgumentParser(prog="Longformer baseline trainer")
parser.add_argument("-d", "--data", choices=["ellie", "participant"], default="participant",
                    help="Data to use")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-1, help="Learning rate")
parser.add_argument("-s", "--seed", type=int, default=42, help="Seed")
args = parser.parse_args()

SEED = args.seed
TARGET_METRIC = "f1-score"  # as in classification_report
if args.data == "ellie":
    DATASET = ("data/AVEC_16_data/train_Ellie.txt", "data/AVEC_16_data/dev_Ellie.txt")
else:
    DATASET = ("data/AVEC_16_data/train_Participant.txt", "data/AVEC_16_data/dev_Participant.txt")
DATASET_NAME = os.path.split(DATASET[0])[1][:-4].split("_")[1]
MODEL = "kiddothe2b/longformer-mini-1024"

# Hyperparameter search
HP_N_TRIALS = 80
HP_RANGE_LR = [1e-6, 1e-1]
HP_RANGE_EPOCHS = [1, 10]

LR = args.learning_rate
EPOCHS = 10
BATCH_SIZE = 8
FINETUNE = True
EVALUATION_STEPS = 10

PATH_TRAIN, PATH_DEV = DATASET
PATH_TEST = ''

dataset_type = re.match(r".+?_(.+).txt", os.path.split(PATH_DEV)[1]).group(1)
PATH_OUTPUT = f"output/{dataset_type}/baseline/bs{BATCH_SIZE}/{DATASET_NAME}/"
PATH_OUTPUT_MODEL =f"model/{dataset_type}/baseline/bs{BATCH_SIZE}/{DATASET_NAME}/"
STUDY_NAME = f"{MODEL}[{'fine-tuned' if FINETUNE else 'pre-trined'}]"
os.makedirs(PATH_OUTPUT, exist_ok=True)
os.makedirs(PATH_OUTPUT_MODEL, exist_ok=True)
OPTUNA_STORAGE = f"sqlite:///{PATH_OUTPUT}db.sqlite3"

training_arguments = {
    "output_dir":PATH_OUTPUT,
    # "learning_rate":None,
    "learning_rate":LR,
    # "num_train_epochs":None,
    "num_train_epochs":EPOCHS,
    "per_device_train_batch_size":BATCH_SIZE,
    "per_device_eval_batch_size":BATCH_SIZE,
    "warmup_steps":100,
    "weight_decay":0.01,
    "evaluation_strategy":"steps",
    "save_strategy":"steps",
    "load_best_model_at_end":True,
    "metric_for_best_model":TARGET_METRIC,
    "save_total_limit":1,
    "eval_steps":EVALUATION_STEPS,
    "seed":SEED
}


np.random.RandomState(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    DEVICE = torch.device('cpu')
print("Device:", DEVICE)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        loss = loss_fct(logits.view(-1, CLASS_WEIGHTS.shape[0]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomMetric(datasets.Metric):
    def __init__(self, metric:str) -> None:
        super(CustomMetric, self).__init__()
        self._metric = metric

    def _info(self):
        return datasets.MetricInfo(
            description='',
            citation='',
            inputs_description='',
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, average="macro", sample_weight=None):
        report = classification_report(references, predictions, output_dict=True)
        if self._metric == 'accuracy':
            value = report[self._metric]
        else:
            value = report[f"{average} avg"][self._metric]
        return {self._metric: value}


# --- auxiliary functions ---

def normalize_text(text):
    # removing punctuation marks to match ASR output format
    # return ' '.join(re.findall(r"\w+", re.sub(r"(<.+?>)|(\*+)", '', text)))
    return text


def tokenize(x_dataset):
    return tokenizer(x_dataset, truncation=True, padding=True)


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))
    print("Model loaded:", MODEL)
    print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, param in model.named_parameters():
        if 'classifier' in name:
            print(f"Size of classification linear layer ('{name}'):", param.shape)

    if not FINETUNE:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


# --- hyperparameter search ---

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate",
                                             HP_RANGE_LR[0], HP_RANGE_LR[1],
                                             log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs",
                                              HP_RANGE_EPOCHS[0], HP_RANGE_EPOCHS[1],
                                              log=True),
    }

def compute_objective(metric):
    # Instead of returning the current value, since there's an issue with optuna (https://github.com/optuna/optuna/issues/2575)
    # that reports the last value of a trial instead of the best intermediate value, I'll always report the best value.
    # So instead of simply:
    # return metric[f"eval_{TARGET_METRIC}"]
    # I'll do, as a workaround (and I'll disable Pruning too):
    global best_metric

    current_metric = metric[f"eval_{TARGET_METRIC}"]
    if (TARGET_METRIC == "loss" and current_metric < best_metric) or \
       (TARGET_METRIC != "loss" and current_metric > best_metric):
        best_metric = current_metric

    return best_metric


metric = CustomMetric(TARGET_METRIC)  # evaluate.load(TARGET_METRIC)
best_metric =  float("inf") if TARGET_METRIC == "loss" else float("-inf")  # https://github.com/optuna/optuna/issues/2575

print("Loading dataset...")
X_train, y_train, ix2label, label2ix = load_dataset(PATH_TRAIN)
X_dev, y_dev, _, _ = load_dataset(PATH_DEV, label2ix=label2ix)
if PATH_TEST:
    X_test, y_test, _, _ = load_dataset(PATH_TEST, label2ix=label2ix)
labels = [ix2label[ix] for ix in range(len(ix2label))]


CLASS_WEIGHTS = torch.tensor(compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=np.array(y_train)), dtype=torch.float)
CLASS_WEIGHTS = CLASS_WEIGHTS.to(DEVICE)

print("Dataset loaded: ")
print(f"    - Training set: {len(y_train)} instances")
print(f"    - Evaluation set: {len(y_dev)} instances")
if PATH_TEST:
    print(f"    - Test set: {len(y_test)} instances")

print("Loading tokenizer and tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
train_dataset = MyDataset(tokenize(X_train), y_train)
dev_dataset = MyDataset(tokenize(X_dev), y_dev)
if PATH_TEST:
    test_dataset = MyDataset(tokenize(X_test), y_test)

trainer = BalancedTrainer(
    # model=model,
    model_init=model_init,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    args=TrainingArguments(**training_arguments),
)

best_trial = trainer.hyperparameter_search(
    hp_space=optuna_hp_space,
    compute_objective=compute_objective,
    n_trials=HP_N_TRIALS,
    direction="minimize" if TARGET_METRIC == "loss" else "maximize",
    backend="optuna",
    storage=OPTUNA_STORAGE,
    study_name=STUDY_NAME,
    load_if_exists=True,
    pruner=optuna.pruners.NopPruner()
)

print(best_trial)

study = optuna.create_study(study_name=STUDY_NAME,
                            storage=OPTUNA_STORAGE,
                            load_if_exists=True)
study.set_user_attr("model", MODEL)
study.set_user_attr("dataset", DATASET)
study.set_user_attr("fine tune", FINETUNE)
# study.set_user_attr(f"best {TARGET_METRIC}", best_trial.objective)
study.set_user_attr(f"best {TARGET_METRIC}", study.best_value)

for arg in best_trial.hyperparameters:
    training_arguments[arg] = best_trial.hyperparameters[arg]

del trainer

training_arguments["save_steps"] = EVALUATION_STEPS
trainer = BalancedTrainer(
    # model=model,
    model_init=model_init,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    args=TrainingArguments(**training_arguments),
)

print("Training...")
trainer.train()

print("EVAL")
val_preds_logits, val_labels , _ = trainer.predict(dev_dataset)
val_preds = np.argmax(val_preds_logits, axis=-1)
print(f"Results for {STUDY_NAME}")
print(classification_report(val_labels, val_preds, target_names=labels, digits=3))

with open(os.path.join(PATH_OUTPUT, "best_results_predictions.json"), "w") as writer:
    json.dump({"y_pred": val_preds.tolist(), "y_true": val_labels.tolist()}, writer)

if PATH_TEST:
    print("TEST SET")
    test_preds_logits, test_labels , _ = trainer.predict(test_dataset)
    test_preds = np.argmax(test_preds_logits, axis=-1)
    print(f"Results for {STUDY_NAME}")
    print(classification_report(test_labels, test_preds, target_names=labels, digits=3))
