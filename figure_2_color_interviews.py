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
import re
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(prog="Generate HTML visualizations for each interview")
parser.add_argument("-d", "--daic-woz",
                    help="Path to the original DAIC-WOZ dataset containing the `<ID>_P/` interview folders")  # f"{oid}_P/{oid}_TRANSCRIPT.csv")
parser.add_argument("-ids", "--target-ids", default="data/AVEC_16_data/dev_IDS.txt",
                    help="File containing the IDS of the interviews to be converted to HTML visualization files")
parser.add_argument("-w-e", "--learned-words-ellie", default="output/Ellie/13_induct-gcn[original-features-1]/words.csv",
                    help="File containing (word, label, weight) triplets")
parser.add_argument("-w-p", "--learned-words-patient", default="output/Participant/21_induct-gcn[original-features250]/words.csv",
                    help="File containing (word, label, weight) triplets")
parser.add_argument("-s", "--speaker", choices=["participant", "ellie", "both"], default="ellie", help="Speaker to highlight")
parser.add_argument("-nw", "--no-highlight-words", action="store_true", help="Disable words highlighting")
parser.add_argument("-o", "--output-folder", default="plots/html",
                    help="Path to the folder to save all the html files for the interviews")
args = parser.parse_args()


ELLIE_COLOR = "#fde725"
PARTICIPANT_COLOR = "#2196f3"
BACKGROUND_COLOR = "#440154"
TXT_COLOR = "white"
KEYWORDS_COLOR = "#fde725"
KEYWORDS_STYLE = f"text-decoration-line: underline; text-decoration-color: {KEYWORDS_COLOR};"

VIRIDIS_CMAP = [
    "#fde72580",
    "#b5de2b80",
    "#6ece5880",
    "#35b77980",
    "#1f9e8980",
    "#26828e80",
    "#31688e80",
    "#3e498980",
    "#48287880",
    "none",  # "#44015480" "none"
][::-1]


def preprocess_text(text):
    text = re.sub(r"\b(?:\w_)+\w\b", lambda m: m.group(0).replace("_", "").upper(), text)
    text = re.sub(r"\b(?:\w+_)*\w+ \((.*)\)$", r"\1", text)
    return re.sub(r"<.+?>", "", text)


pwords = pd.read_csv(args.learned_words_patient, header=None, names=["word", "label", "value"])
ewords = pd.read_csv(args.learned_words_ellie, header=None, names=["word", "label", "value"])
pwords = pwords[pwords.label == "positive"]
ewords = ewords[ewords.label == "positive"]
words = {
    "participant": pwords.word.values.tolist(),
    "ellie": ewords.word.values.tolist()
}

ids = pd.read_csv(args.target_ids, sep="\t")

os.makedirs(args.output_folder, exist_ok=True)
for _, interview in ids.iterrows():
    oid = interview.original_ID
    category = "depressed" if interview.category else "control"
    html_filename = f"{category}_{'no_words_' if args.no_highlight_words else ''}{oid}.html"

    df = pd.read_csv(os.path.join(args.daic_woz, f"{oid}_P/{oid}_TRANSCRIPT.csv"), sep="\t")
    df.fillna("", inplace=True)
    df.value = df.value.map(preprocess_text)
    df.speaker = df.speaker.str.lower()

    keywords_map = np.zeros(len(df))
    speaker_map = np.array([int(spkr == "ellie") for spkr in df.speaker])
    for ix, row in df.iterrows():
        keywords_map[ix] = len([w for w in row.value.split() if w in words[row.speaker]])

    if speaker_map.sum() > 0 and keywords_map[speaker_map == 1].max() != 0:
        keywords_map[speaker_map == 1] /= keywords_map[speaker_map == 1].max()
    if keywords_map[speaker_map != 1].max() != 0:
        keywords_map[speaker_map != 1] /= keywords_map[speaker_map != 1].max()

    html_filename = os.path.join(args.output_folder, html_filename)
    with open(html_filename, "w") as writer:  # TODO: ADD PREDICTION TO FILENAME
        writer.write('<link rel="stylesheet" type="text/css" href="css/materialize.min.css" />\n')
        writer.write(f'<body style="color: {TXT_COLOR}; background-color: {BACKGROUND_COLOR}">\n')
        for ix, row in df.iterrows():
            bcolor = ELLIE_COLOR if row.speaker == "ellie" else PARTICIPANT_COLOR
            tcolor = "none"
            text = row.value
            if args.speaker == "both" or row.speaker == args.speaker:
                text = " ".join([f'<span style="{KEYWORDS_STYLE}">{w}</span>' if not args.no_highlight_words and w in words[row.speaker] else w
                                for w in text.split()])
                tcolor = VIRIDIS_CMAP[int(keywords_map[ix] * (len(VIRIDIS_CMAP) - 1))]
            writer.write(f'<div class="row" style="margin-bottom: 0; background-color: {tcolor}"><b style="color: {bcolor}">{row.speaker.title()}:</b> <span>{text}</span></div>\n')
        print(f"Interview {oid} saved in `{html_filename}`")
