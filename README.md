# Text classification: easy implementation, BERT fine tuning, BERT Distillation & TF Hub encoder

In this repo, I constructed a quick overview/tutorial to classify texts. We go over a few methods motivated by an article on [BERT distillation with spaCy](https://towardsdatascience.com/distilling-bert-models-with-spacy-277c7edc426c) and we are going through the three steps of the process:

1. Create a [text classifier](https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py) algorithm with spaCy
2. [Fine tune BERT](https://github.com/google-research/bert#fine-tuning-with-bert) to create a more efficient classifier
3. [Distill](https://arxiv.org/pdf/1503.02531.pdf) the BERT algorithm to get an efficient model but lighter than the BERT

I also computed the results with the help of [TensorFlow Hub](https://tfhub.dev/). You can connect to this hub and load a [large universal encoder](https://tfhub.dev/google/universal-sentence-encoder-large/4). With this encoder you can encode each of your sentence in a vector of 512 features. And with it add a classifier from [Scikit-learn](https://scikit-learn.org) to create a text classifier in a few lines of codes. Here I used a gradient boosting classifier.

The results are for the [dataset](https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv) I collected from [kaggle](https://www.kaggle.com/) and for a [hatred speech dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) I found on github. Nevertheless, you can use this code to perform you own text classification with any dataset in `data/` where the `X_train.npy` and `X_test.npy` are arrays of text and `y_train.py` and `y_test.npy` are arrays of label (with as many label as you want). To perform this go directly to the [*spaCy textcat implementation*](https://github.com/agombert/textcat/blob/master/README.md#spacy-textcat-implementation).

## Main results

### Wine reviews

Our [train dataset](https://github.com/agombert/textcat/data/Text_train.npy) has a length 4000. It's balanced between the four classes. The [test dataset](https://github.com/agombert/textcat/data/Text_test.npy) has a length 4000. Thus we will be able to see the difference with the transfert learning method (BERT fine tuning).

For each algorithm I trained for 15 epochs with batch size 32 and a dropout at 0.5. 

I computed the accuracy and the macro recall / macro precision for each model.

|      Model     | Accuracy | Recall | Precision | Model Size|
|:--------------:|:--------:|:------:|:---------:|:---------:|
| *spaCy CNN*    |  47.45%  | 47.45% |   52.38%  |   5.3Mb   |
|*BERT FineTuned*|  58.03%  | 58.03% |   61.15%  |   1.2Gb   |
|*Distilled BERT*|  51.85%  | 51.85% |   53.29%  |   6.2MB   | 

I also computed the recall/precision for each class. 

|      Model     | Recall 0 | Precision 0 | Recall 1 | Precision 1 | Recall 2 | Precision 2 | Recall 3 | Precision 3 |
|:--------------:|:--------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|
|*spaCy CNN*     |58.30%|35.55%|35.50%|39.10%|46.30%|75.78%|49.70%|59.10%|
|*BERT FineTuned*|65.40%|80.54%|55.60%|49.12%|56.20%|43.74%|54.90%|71.21%|
|*Distilled BERT*|46.40%|39.36%|43.20%|43.03%|59.40%|70.38%|58.40%|60.02%|

### Hatre Speech detection

Our [train dataset](https://github.com/agombert/textcat/data/HS_X_train.npy) has a length 2145. It's balanced between the three classes. The [test dataset](https://github.com/agombert/textcat/data/HS_X_test.npy) has a length 2145. Thus we will be able to see the difference with the transfert learning method (BERT fine tuning).

For each algorithm I trained for 15 epochs with batch size 32 and a dropout at 0.5. 

I computed the accuracy and the macro recall / macro precision for each model.

|      Model     | Accuracy | Recall | Precision | Model Size|
|:--------------:|:--------:|:------:|:---------:|:---------:|
| *spaCy CNN*    |  59.91%  | 59.91% |   60.52%  |   5.3Mb   |
|*BERT FineTuned*|  81.35%  | 81.35% |   81.58%  |   1.2Gb   |
|*Distilled BERT*|  72.73%  | 72.73% |   72.35%  |   6.2MB   |
|*TFH Encod + GB*|  77.02%  | 77.02% |   76.88%. |   0.5MB*  |

I also computed the recall/precision for each class. 

|      Model     | Recall 0 | Precision 0 | Recall 1 | Precision 1 | Recall 2 | Precision 2 |
|:--------------:|:--------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|
|*spaCy CNN*     |58.74%|53.30%|61.68%|69.78%|59.30%|58.48%|
|*BERT FineTuned*|90.34%|89.34%|75.25%|82.14%|78.60%|73.27%|
|*Distilled BERT*|60.14%|68.58%|87.69%|77.22%|70.35%|71.25%|
|*TFH Encod + GB*|86.85%|81.39%|76.50%|74.22%|67.69%|75.04%|

* the size of the model does not take into account the encoder as it has been loaded from the hub and is not saved.


## Data

### Wine dataset

For this experiment I used a [dataset of wine reviews](https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv) I found on Kaggle. I cut out the dataset depending on the points one gave to a wine. The association label-mark is derived from the quartiles:

| Label | mark min | mark max |
|:-----:|:--------:|:--------:|
| 0     | 80       | 85       |
| 1     | 86       | 87       |
| 2     | 88       | 90       |
| 3     | 91       | 100      |

The objective is according to the text, rank the review. In term or real like applications, we can use those kind of algorithms to detect hatred speech on any forum or twitter. We can also use those algorithms to assess the global opinion on a particular subject (movies, policies, branding...). It is classic sentiment analysis. 

Here the difference is that we look at sentiment analysis with more than two labels (0-1) that we can easily find on tutorials.

### Hatred speech dataset

For this experiment I used a [dataset of hatred speech](https://github.com/t-davidson/hate-speech-and-offensive-language) I found on Github. The speech is already labelled with 0 for hatred speech, 1 for offensive speech and 2 for neutral speech. I only managed to anonymized the tweets when there are some mention within the tweet.

|   Class  | Label |
|:--------:|:-----:|
| Hatred   | 2     |
| Offensive| 1     |
| Neutral  | 0     |

The objective is according to the text, rank the tweet.

## spaCy textcat implementation

When you have your dataset with the texts and labels you can use `run_textcat.py` to make the classification and save your model.

Save your train test in `data/` with a name followed by the mention `_train` with a `.npy` format (with numpy.save) and your test set with the mention `_test` with the same extention `.npy`.


```bash
python3 run_textcat.py --is predict False
                       --name_model name_of_model
                       --cats cat1 cat2 cat3 ...
                       --X name of features data (without _train)
                       --y name of labels data (without _train)
                       --bs_m           The minimum batch size for training
                       --bs_M           The maximum batch size for training
                       --step           The step to go from min batchsize to max batchsize
                       --epoch          Number of epoch for the training
                       --drop           Drop out to apply the model
```

Your model will be save in `models/` with one file `name_of_model_nlp` which is the spaCy model associated, `name_of_model_textcat` which is the spacy Textcomponent component and `name_of_models_scores` which are the scores on the evaluation dataset (automatically created) during training.

Then you can use the same function to get metrics on the test dataset:

```bash
python3 run_textcat.py --is predict True
                       --name_model name_of_model
                       --X name of features data (without _test)
                       --y name of labels data (without _test)
```

And you'll get the evaluations in logs. 

## BERT textcat implementation (with google Colab)

For this part I used google colab, as it's really cool to get free GPU access and perform BERT fine tuning on small datasets. I I used the code from this [colab](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=dsBo6RCtQmwx) which uses BERT to review movies. I aranged a bit the code to adapt it to my problem. 

The code is [here](https://colab.research.google.com/drive/1MShG1gDV5TfvVEYDTBgr5LCzVNVmTf03#scrollTo=NwW9OH0CBJx9&uniqifier=1), and adapting to your problem you can use it for any text classification by fine tuning BERT. 

You just follow each cell, but previously you'll need some storage ([GCS](https://console.cloud.google.com/storage) for instance) to store the results and your datasets. 

## BERT distillation implementation

### [Data Augmentation](https://arxiv.org/abs/1503.02531)

For the augmented data we follow two out of three methods from [Distilling Task-Specific Knowledge from BERT intoSimple Neural Networks](https://arxiv.org/pdf/1903.12136.pdf). We mask some tokens and also change the order of randomly chosen n-grams in the sentence. At the end we go from 4k to 200k data for our augmented set.

You can use a [google colab notebook](https://colab.research.google.com/drive/128apJ8WAMVyXxocCY9CRapsr1Qs8Mu0w#scrollTo=zXkPH_rUatS6) once again or you can do it in local and stock your augmented data in the bucket you previously created. 

You can also perform this code locally to get 50 times more data. Then we will use the previously trained model to compute the prevision for the augmented data and also the probabilities associated at each text. 

### Training on augmented data

When you have your augmented data and also the BERT predictions on those data (labels and probabilities) we are going to train a new model from spaCy on those data.

First we use only the labels predicted by the BERT fineted model. So we use exactly the same method as in the section on spaCy textcat. 

We outperformed first model by 13% on hatred speech datasets and 4.5% on the wine datasets ! We still are below the BERT from a significagive margin, but we should add the loig probs to the loss function to try to improve the model. And the size of the model is a bit higher from the first textcat spaCy classifier we had. 

## TFHub encoder + Gradient Boosting

I was at first wondering how to modify the spaCy loss function to make the textcat algorithm works like a regression on each category. Instead of that I tried to use an encoder from TFHub. 

The [encoder](https://tfhub.dev/google/universal-sentence-encoder-large/4) I used for is well explained on the hub. 

`
The input is variable length English text and the output is a 512 dimensional vector. The universal-sentence-encoder-large model is trained with a Transformer encoder
`

Thus by using the encoder on each sentence, without finetuning or new training to focus on our data we can try to classify the text. 

It looks really good as we outperform all the models except the BERT and we are not so far as we are only 4.5% below for the hatred speech and X% for the wine classification.
