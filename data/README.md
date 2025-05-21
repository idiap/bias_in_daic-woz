### Data preparation

The `AVEC_16_data` contains all the needed files for training en evaluating the models and replicating the results reported in the paper.

Training and development set files are expected to be named `SPLIT_SPEAKER.txt` (e.g. `train_Ellie.txt`, `dev_Participant.txt`) and must contain one interview per line. Depending on the speaker, each file line (interview) should be the concatenation of all the utterances in the interview of the specified speaker only (i.e. only Ellie or Participant's utterances). Finally, each line shuld contain not only the interview but also the its ground truth label separated by tab character, as in the following example (`train_Participant.txt`):

```tsv
negative	okay how 'bout yourself here in california yeah oh well...
negative	i'm doing good um from los angeles california um the co...
...
```

> ⚠️ **NOTE:** Due to **license restrictions** of the DAIC-WOZ datasets, only metada files are fully provided in [`AVEC_16_data`](AVEC_16_data/) folder. The needed training and development set files ([`train_Ellie.txt`](AVEC_16_data/train_Ellie.txt), [`dev_Ellie.txt`](AVEC_16_data/dev_Ellie.txt), [`dev_Participant.txt`](AVEC_16_data/dev_Participant.txt), [`train_Participant.txt`](AVEC_16_data/train_Participant.txt) files) inside the folder only contain two fake examples as an example of the expected format. Please **download the original datasets from [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)**, convert the dataset in the expected txt files (tab-separated formatted as the descrived above) for each speaker, and replace the 4 training and development txt files that are inside these folders.
