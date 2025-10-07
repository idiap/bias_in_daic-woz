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
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

CMAP = "viridis"  # "magma"
TARGET_INTERVIEW = 381  # to highlight with the red rectangle
PATH_DEV_IDS = "data/AVEC_16_data/dev_IDS.txt"  # only used to map the above ID to the plot column index

parser = argparse.ArgumentParser(prog="Plot temporal heatmap")
parser.add_argument("-train-e", "--interviews-train-ellie", default="data/AVEC_16_data/train_Ellie.txt",
                    help="File contianing the training interviews and the labels")
parser.add_argument("-dev-e", "--interviews-dev-ellie", default="data/AVEC_16_data/dev_Ellie.txt",
                    help="File contianing the development interviews and the labels")
parser.add_argument("-test-e", "--interviews-test-ellie", default=None,
                    help="File contianing the test interviews and the labels")
parser.add_argument("-train-p", "--interviews-train-patient", default="data/AVEC_16_data/train_Participant.txt",
                    help="File contianing the training interviews and the labels")
parser.add_argument("-dev-p", "--interviews-dev-patient", default="data/AVEC_16_data/dev_Participant.txt",
                    help="File contianing the development interviews and the labels")
parser.add_argument("-test-p", "--interviews-test-patient", default=None,
                    help="File contianing the test interviews and the labels")
parser.add_argument("-w-e", "--learned-words-ellie", default="output/Ellie/13_induct-gcn[original-features-1]/words.csv",
                    help="File containing (word, label, weight) triplets")
parser.add_argument("-w-p", "--learned-words-patient", default="output/Participant/21_induct-gcn[original-features250]/words.csv",
                    help="File containing (word, label, weight) triplets")
parser.add_argument("-o", "--output-plot", default="temporal_heatmap.png",
                    help="PNG file to store the plot")
parser.add_argument("-t", "--top-k", type=int, default=0, help="Top-k words to consider")
parser.add_argument("-b", "--bins", type=int, default=10,
                    help="Number of bins to use in the histograms")
parser.add_argument("-all", "--show-all", action="store_true", help="Show plots for all interviews")
parser.add_argument("-q", "--quiet", action="store_true", help="Do not show plots")
parser.add_argument("-k", "--kde-bandwidth", type=float, default=.05, help="KDE bandwidth")
parser.add_argument("-s", "--seed", type=int, default=13, help="Seed")
args = parser.parse_args()
np.random.seed(args.seed)

try:
    dev_ids = pd.read_csv(PATH_DEV_IDS, sep="\t")
    target_interview_ix = dev_ids[dev_ids.original_ID == TARGET_INTERVIEW].row_ID.values[0]
except:
    target_interview_ix = -1

interviews_pos, interviews_neg = {}, {}
interviews_pos_x, interviews_neg_x = {}, {}
train_n, dev_n, words = {}, {}, {}
for spkr in ["p", "e"]:
    train_set = args.interviews_train_ellie if spkr == "e" else args.interviews_train_patient
    dev_set = args.interviews_dev_ellie if spkr == "e" else args.interviews_dev_patient
    test_set = args.interviews_test_ellie if spkr == "e" else args.interviews_test_patient

    df = pd.read_csv(train_set, sep="\t", header=None)
    df.fillna("", inplace=True)
    interviews_pos[spkr] = [d.split() for d in df[df[0] == "positive"][1].values]
    interviews_neg[spkr] = [d.split() for d in df[df[0] == "negative"][1].values]
    # np.random.shuffle(interviews_pos)
    # np.random.shuffle(interviews_neg)
    train_n[spkr] = [len(interviews_pos[spkr]), len(interviews_neg[spkr])]

    df = pd.read_csv(dev_set, sep="\t", header=None)
    df.fillna("", inplace=True)
    df["mark"] = False
    if target_interview_ix != -1:
        df.loc[target_interview_ix - 1, "mark"] = True
    if spkr == "e":
        interviews_mark = len(interviews_pos[spkr]) + np.where(df[df[0] == "positive"].mark.values == True)[0][0]
    interviews_pos[spkr] += [d.split() for d in df[df[0] == "positive"][1].values]
    interviews_neg[spkr] += [d.split() for d in df[df[0] == "negative"][1].values]
    dev_n[spkr] = [len(interviews_pos[spkr]), len(interviews_neg[spkr])]

    if test_set is not None:
        df = pd.read_csv(test_set, sep="\t", header=None)
        df.fillna("", inplace=True)
        interviews_pos[spkr] += [d.split() for d in df[df[0] == "positive"][1].values]
        interviews_neg[spkr] += [d.split() for d in df[df[0] == "negative"][1].values]

    df = pd.read_csv(args.learned_words_ellie, header=None)
    df = df[df[1] == "positive"]

    words[spkr] = [(row[0], row[2]) for _, row in df.iterrows()]
    words[spkr] = [w for w, v in sorted(words[spkr], key=lambda wv: -wv[1])][:args.top_k if args.top_k > 0 else None]

    interviews_pos[spkr] = [[int(w in words[spkr]) for w in d] for d in interviews_pos[spkr]]
    interviews_neg[spkr] = [[int(w in words[spkr]) for w in d] for d in interviews_neg[spkr]]

    interviews_pos_x[spkr] = [np.where(np.array(interview) == 1)[0] / len(interview)
                              for interview in interviews_pos[spkr]]
    interviews_neg_x[spkr] = [ix / len(interview) for interview in interviews_neg[spkr]
                              for ix in np.where(np.array(interview) == 1)]

titles = ["Depressed", "Control"]
ylabels = ["$E$-GCN", "$P$-GCN"]
fig, axs = plt.subplots(2, 2, sharey=True, sharex="col")
for spkr_ix, spkr in enumerate(["e", "p"]):
    for ax_ix, interviews in enumerate([interviews_pos_x[spkr], interviews_neg_x[spkr]]):
        n_interviews = len(interviews)
        X_plot = np.linspace(0, 1, 100)
        kdes = np.zeros((X_plot.shape[0], n_interviews))
        X_plot_sample = X_plot[:, np.newaxis]
        for ix, interview in enumerate(interviews):
            X = np.array(interview)[:, np.newaxis]
            if X.shape[0] > 0:
                kde = KernelDensity(kernel="gaussian", bandwidth=args.kde_bandwidth).fit(X)
                dens = np.exp(kde.score_samples(X_plot_sample))
                kdes[:, ix] = dens

        x = np.arange(n_interviews+ 1) + 1
        y = np.linspace(0, 1, 101)
        z = np.flip(kdes, axis=0)

        z_min = kdes.min()
        z_max = kdes.max()

        ax = axs[spkr_ix, ax_ix]
        if ax_ix == 0:
            ax.set_ylabel(ylabels[spkr_ix])
        if spkr_ix == 0:
            ax.set_title(titles[ax_ix])
        if spkr_ix == 1:
            ax.set_xlabel("Interviews")
        ax.pcolormesh(x, y, z, cmap=CMAP, vmin=z_min, vmax=z_max)
        ax.set_ylim(0, 1)
        y_ticks = [0, .25, .5, .75, 1]
        ax.yaxis.set_ticks(y_ticks)
        ax.set_yticklabels([f"{1 - v:.0%}" for v in y_ticks])
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)

        ax.vlines(train_n[spkr][ax_ix] + 1, colors="white", ymin=0, ymax=1, linewidth=2)
        ax.vlines(dev_n[spkr][ax_ix] + 1, colors="white", ymin=0, ymax=1, linewidth=2)

        if spkr == "e" and ax_ix == 0:
            y_max = 1 - np.argmax(kdes[:, interviews_mark]) / 100
            height = .06
            ax.add_patch(plt.Rectangle((interviews_mark + 1, y_max - height / 2),
                                       1, height,
                                       linewidth=2, edgecolor="red",
                                       fill=False, alpha=.75))
            # ax.vlines(interviews_mark + 1, colors="red", ymin=y_max - .1, ymax=y_max + .1, linewidth=1)

plt.tight_layout()
plt.savefig(args.output_plot, dpi=300)
if not args.quiet:
    plt.show()
