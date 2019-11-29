# Text classification: easy implementation, BERT fine tuning and BERT Distillation

In this repo, I constructed a quick overview/tutorial to classify texts. We go over a few methods motivated by an article on [BERT distillation with spaCy](https://towardsdatascience.com/distilling-bert-models-with-spacy-277c7edc426c) and we are going through the three steps of the process:

1. Create a [text classifier](https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py) algorithm with spaCy
2. [Fine tune BERT](https://github.com/google-research/bert#fine-tuning-with-bert) to create a more efficient classifier
3. [Distill](https://arxiv.org/pdf/1503.02531.pdf) the BERT algorithm to get an efficient model but lighter than the BERT

The results are for the [dataset](https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv) I collected from [kaggle](https://www.kaggle.com/). Nevertheless, you can use this code to perform you own text classification with any dataset in `data/` where the `Text_train.npy` and `Text_test.npy` are arrays of text and `Cat_train.py` and `Cat_test.npy` are arrays of label (with as many label as you want). To perform this go directly to the [*spaCy textcat implementation*](https://github.com/agombert/textcat/blob/master/README.md#spacy-textcat-implementation)

## Main results

Our [train dataset](https://github.com/agombert/textcat/data/Text_train.npy) has a length 53920. It's balanced between the four classes. The [test dataset](https://github.com/agombert/textcat/data/Text_test.npy) has a length 23104. 

For each algorithm I trained for 10 epochs with batch size 32 and a dropout at 0.5. 

I computed the accuracy and the macro recall / macro precision for each model.

|      Model     | Accuracy | Recall | Precision | Model Size|
|:--------------:|:--------:|:------:|:---------:|:---------:|
| *spaCy CNN*    |  59.99%  | 59.99% |   59.00%  |   5.3Mb   |
|*BERT FineTuned*|        |      |           |           |
|*Distilled BERT*|        |      |           |           |

I also computed the recall/precision for each class. 

|      Model     | Recall 0 | Precision 0 | Recall 1 | Precision 1 | Recall 2 | Precision 2 | Recall 3 | Precision 3 |
|:--------------:|:--------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|
|*spaCy CNN*     |  38.09%  |    47.38%   |  45.45%. |   50.40%    |  74.43%  |   75.69%    |  82.01%  |   62.55%    |
|*BERT FineTuned*|        |      |           |           |
|*Distilled BERT*|        |      |           |           |

## Data

For this experiment I used a [dataset of wine reviews](https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv) I found on Kaggle. I cut out the dataset depending on the points one gave to a wine. The association label-mark is derived from the quartiles:

| Label | mark min | mark max |
|:-----:|:--------:|:--------:|
| 0     | 80       | 85       |
| 1     | 86       | 87       |
| 2     | 88       | 90       |
| 3     | 91       | 100      |

The objective is according to the text, rank the review. In term or real like applications, we can use those kind of algorithms to detect hatred speech on any forum or twitter. We can also use those algorithms to assess the global opinion on a particular subject (movies, policies, branding...). It is classic sentiment analysis. 

Here the difference is that we look at sentiment analysis with more than two labels (0-1) that we can easily find on tutorials.

## spaCy textcat implementation

When you have your dataset with the texts and labels you can use `run_textcat.py` to make the classification and save your model.

Save your train test in `data/` with a name followed by the mention `_train` with a `.npy` format (with numpy.save) and your test set with the mention `_test` with the same extention `.npy`.


```python
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

```python
python3 run_textcat.py --is predict True
                       --name_model name_of_model
                       --X name of features data (without _test)
                       --y name of labels data (without _test)
```

And you'll get the evaluations in logs. 

## BERT textcat implementation (with google Colab)

## BERT distillation implementation

### Data Augmentation

### Training on augmented data
