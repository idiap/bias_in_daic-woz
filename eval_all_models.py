# -*- coding: utf-8 -*-
"""
Script for running the InducT-GCN experiments with optuna optimization.

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
import json
import argparse

from sklearn.metrics import classification_report


parser = argparse.ArgumentParser(prog="Generate Results including with ensemble of best models")
parser.add_argument("-i", "--path-results", default="output", help="Folder containing the experiments results")
parser.add_argument("-r", "--classification-report", action="store_true", help="Show classification report")
args = parser.parse_args()

CR_DIGITS = 2

def pred_ensemble(results, model_a, model_b):
    pred_model_a = results[model_a]["predictions"]["y_pred"]
    pred_model_b = results[model_b]["predictions"]["y_pred"]

    assert results[model_a]["predictions"]["y_true"] == results[model_b]["predictions"]["y_true"]
    return [int(y_a and pred_model_b[ix]) for ix, y_a in enumerate(pred_model_a)]


best_models = {}
for folder in os.listdir(args.path_results):
    path_experiment = os.path.join(args.path_results, folder)
    if not os.path.isdir(path_experiment):
        continue
    
    best_models[folder] = {
        "score": float("-inf"),
        "results": None,
        "predictions": None,
        "model": None,
        "words": None
    }

    for root, dirs, files in os.walk(path_experiment):
        if "results.json" not in files:
            continue

        with open(os.path.join(root, "results.json")) as reader:
            results = json.load(reader)

        score = results['macro avg']['f1-score']
        print(f"{root}: {score:.2%}")
        if score > best_models[folder]["score"]:
            with open(os.path.join(root, "results_predictions.json")) as reader:
                predictions = json.load(reader)
            best_models[folder]["score"] = score
            best_models[folder]["results"] = results
            best_models[folder]["predictions"] = predictions
            best_models[folder]["model"] = root
            best_models[folder]["words"] = [fl for fl in files if "words.csv" in fl][0]

print()
y_true = best_models["Participant"]["predictions"]["y_true"]
for dataset in best_models:
    path_words = os.path.join(best_models[dataset]["model"], best_models[dataset]["words"])
    path_plot = best_models[dataset]["model"]
    print(f"Best result for '{dataset}': {best_models[dataset]['score']:.2%} ({best_models[dataset]['model']})")
    if args.classification_report:
        print(classification_report(y_true, best_models[dataset]["predictions"]["y_pred"], digits=CR_DIGITS))
        print()

y_pred_we = pred_ensemble(best_models, "Participant", "Ellie")

score_we = classification_report(y_true, y_pred_we, output_dict=True)['macro avg']['f1-score']

print(f"Result ensemble bests 'Participant' and 'Ellie': {score_we:.2%}")
if args.classification_report:
    print(classification_report(y_true, y_pred_we, digits=CR_DIGITS))
    print()
