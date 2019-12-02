from os import getcwd
from os.path import join
import numpy as np
import pickle
import random
import logging

logging.basicConfig(level=logging.INFO)

import spacy
from spacy.util import minibatch, compounding
from spacy.pipeline import TextCategorizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

PATH_REPO = getcwd()
PATH_DATA = join(PATH_REPO, 'data')

def load_data(name_features, name_labels):
    """
    Objective: Load the data, features in X and labels in y
    
    Inputs:
        - name_features, str: the name of the feature file in the data repo
        - name_labels, str: the name of the label file in the data repo
    Outputs:
        - X, np.array: array of the features for the text (1D with each line a text)
        - y, np.array: array of the labels (1D with each line a label) must be labels of int begining at 0
    """
    X = np.load(join(PATH_DATA, '{}.npy'.format(name_features)), allow_pickle=True)
    y = np.load(join(PATH_DATA, '{}.npy'.format(name_labels)), allow_pickle=True)
    
    logging.info('Data loaded')
    
    return(X, y)


def load_model_labels(*args, **kwargs):
    """
    Objective: Load the spacy models we want, or create a blank one and add the different labels we have with the
                good names in args (and in order !) in kwargs we have some models specifications we can define or 
                change
    
    Inputs:
        - *args, str: the arguments for each label 1st one corresponds to the cat 0, 2nd one to 1,...
        - **kwargs: in kwargs we can give the name of the english model in spacy, we can also give the kwargs
                    architecture of our textcat classifier (look at spacy doc for that)
    Outputs:
        - nlp, spaCy.model: a spaCy model with textcat in the pipe
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model
        - scores, dict: empty dict to get the results during training
    """
    
    model = kwargs.get('model', False)
    
    if model:
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("en")
        
    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True,
                                             "architecture": kwargs.get('model_text_cat', "simple_cnn")})
    
    nlp.add_pipe(textcat, last=True)
    
    for arg in args:
        textcat.add_label(arg)
    
    scores_ = {}
    
    logging.info('New model created')
    
    return(nlp, textcat, scores_)


def preprocess_before_training(textcat, X_train, y_train, r=0.8):
    """
    Objective: before going further we need to preprocess the numpy arrays
    
    Inputs:
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model
        - X_train, np.array: array of the features for the text (1D with each line a text)
        - y_train, np.array: array of the labels (1D with each line a label)
        - r, float: between 0 and 1, the ratio of the total train set to use as train 1-r used as eval set
    Outputs:
        - train_data, list: the list of (text, cat) for training
        - dev_texts, tuple: the texts to assess the training on
        - de_cats, np.array: the labels to assess the training on
    """
    
    assert r < 1 and r >0, "ValueError: r should be between 0 and 1 strictly"
    
    
    ind_train, ind_val = balance_train_val(X_train, y_train, 1-r)
    
    X_train, dev_texts = tuple(X_train[ind_train]), tuple(X_train[ind_val])
    y_train, y_val = y_train[ind_train], y_train[ind_val] 
    
    y_train = [{label: y==i for i, label in enumerate(textcat.labels)} for y in y_train]
    dev_cats = [{label: y==i for i, label in enumerate(textcat.labels)} for y in y_val]
    
    train_data = list(zip(X_train, [{"cats": cats} for cats in y_train]))
    
    
    logging.info('Data preprocessed for training')
    
    return(train_data, dev_texts, dev_cats)


def balance_train_val(X_train, y_train, r):
    """
    Objective: balance train and val datasets
    
    Inputs:
        - X_train, np.array: array of the features for the text (1D with each line a text)
        - y_train, np.array: array of the labels (1D with each line a label)
        - r, float: between 0 and 1, the ratio 1-r for train set to use as train r used as eval set
    Outputs:
        - ind_train, list: the list of the indices to use for training set
        - ind_val, list: the list of the indices to use for validation set
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=r)
    sss.get_n_splits(X_train, y_train)
    
    for train_index, test_index in sss.split(X_train, y_train):
        ind_train, ind_val = train_index, test_index
        
    return(ind_train, ind_val)


def training(nlp, textcat, train_data, dev_texts, dev_cats, n_iter=10, bs_min=4,
             bs_max=32, step=1.001, drop=0.2, verbose=1,
             scores_={}):
    """
    Objective: train the models with some hyperparameters to fix in inputs
    
    Inputs:
        - nlp, spaCy.model: a spaCy model with textcat in the pipe we want to train/update
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model we want to train/update
        - train_data, list: the list of (text, cat) for training
        - dev_texts, tuple: the texts to assess the training on
        - de_cats, np.array: the labels to assess the training on
        - n_iter, int: the number of epochs to train the model
        - bs_min, int: look at compounding in spaCy doc, but the minimum batch size for training
        - bs_max, int: look at compounding in spaCy doc, but the maximum batch size for training
        - step, float: look at compounding in spaCy doc, defines how to grow from bs_min to bs_max
        - drop, float: should be between 0 and 1, the dropout for the data
        - verbose, bool: False no verbosity, otherwise we will have the progress scores
        - scores_, dict: the metrics at each step after training
    Outputs:
        - nlp, spaCy.model: a spaCy model with textcat in the pipe updated
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model updated
        - scores_, dict: the metrics at each step after training
        - losses, dict: the losses during training
    """
    m = len(textcat.labels)
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    
    scores_ = scores_
    n_old = 0 if len(scores_.keys()) == 0  else min(scores_.keys())
    
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if verbose > 0:
            logging.info("Training the model...")
            t = "\t{:^5}"*m
            formats = ['{:^5}\t{:^5}'.format('R{}'.format(i), 'P{}'.format(i)) 
                       for i in range(1, m+1)]

            logging.info("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "Acc", "R", "P") + t.format(*formats))
            
        
        batch_sizes = compounding(bs_min, bs_max, step)
        
        for i in range(n_old, n_old + n_iter):
            losses = {}
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=drop, losses=losses)
                
            
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
                scores_[i] = scores
                scores_[i]['losses'] = losses['textcat']
                if verbose > 0:  
                    t = "\t{:^5}"*m
                    formats2 = ['{:.2f}%\t{:.2f}%'.format(scores["recall_{}".format(i)]*100,
                                                      scores["precision_{}".format(i)]*100)
                                for i in range(1, m+1)]
                    formats1 = [losses["textcat"], scores["acc"]*100, 
                                scores["recall"]*100, scores["precision"]*100]
                    
                    logging.info("{:.2f}\t{:.2f}%\t{:.2f}%\t{:.2f}%".format(*formats1) + t.format(*formats2))
                
    
    return(nlp, textcat, scores_)



def get_results(y_true, y_pred):
    """
    Objective: Once we have the ground truths and predictions we compute the confusion matrix in order to 
                get the micro (for each class) and macro (global) recall/precision we compute also accuracy
    
    Inputs:
        - y_true, list or np.array: the ground truths labels
        - y_pred, list or np.array: the predictions we made
    Outputs:
        - results, dict: with as keys/values the recall/precision (micro and macro) and the accuracy of the model
    """
    cm = confusion_matrix(y_true, y_pred)

    results = {}
    
    m = len(np.unique(y_true))
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    
    results['acc'] = np.sum(TP) / np.sum(cm)

    for i in range(m):
        results['recall_{}'.format(i+1)] = TP[i] / (TP[i] + FN[i])
        results['precision_{}'.format(i+1)] = TP[i] / (TP[i] + FP[i])
    
    results['recall'] = np.mean([results.get('recall_{}'.format(i+1)) for i in range(m)])
    
    results['precision'] = np.mean([results.get('precision_{}'.format(i+1)) for i in range(m)])
        
    return results
    

def evaluate(tokenizer, textcat, texts, cats):
    """
    Objective: evaluate the model on new texts associated with the gold catergories in cat
    
    Inputs:
        - tokenizer, spaCy tokenizer: the tokenizer use in spaCy model
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model
        - texts, tuple: the texts to evaluate textcat on
        - cats, list of dict: the cats to evaluate the textcat on
    Ouputs:
        - results, dict: with as keys/values the recall/precision (micro and macro) and the accuracy of the model
    """
    docs = (tokenizer(text) for text in texts)
    y_true, y_pred = [], []
    for i, doc in enumerate(textcat.pipe(docs)):
        score1 = 0
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score > score1:
                score1 = score
                pred_l = label
            if gold[label]:
                true_l = label

        y_true.append(true_l)
        y_pred.append(pred_l)
    
    results = get_results(y_true, y_pred)
    
    return results


def evaluate_test(X_test, y_test, textcat, nlp):
    """
    Objective: Evaluate the model on a test set
    
    Inputs:
        - X_test, np.array: array of the features for the text (1D with each line a text)
        - y_test, np.array: array of the labels (1D with each line a label)
        - nlp, spaCy.model: a spaCy model with textcat in the pipe updated
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model updated
    Outputs:
        -scores, dict: the dictionary of the results
    """
    m = len(np.unique(y_test))
    
    y_test = [{label: y==i for i, label in enumerate(textcat.labels)} for y in y_test]
    
    X_test = tuple(X_test)
    
    scores = evaluate(nlp.tokenizer, textcat, X_test, y_test)
    t = "\t{:^5}"*m
    formats = ['{:^5}\t{:^5}'.format('R{}'.format(i), 'P{}'.format(i)) 
               for i in range(1, m+1)]

    logging.info("{:^5}\t{:^5}\t{:^5}".format("Acc", "R", "P") + t.format(*formats))
            
    formats2 = ['{:.2f}%\t{:.2f}%'.format(scores["recall_{}".format(i)]*100,
                                      scores["precision_{}".format(i)]*100)
                for i in range(1, m+1)]
    formats1 = [scores["acc"]*100, 
                scores["recall"]*100, scores["precision"]*100]

    logging.info("{:.2f}%\t{:.2f}%\t{:.2f}%".format(*formats1) + t.format(*formats2))
    
    return(scores)


def save_results(name, nlp, textcat, scores_):
    """
    Objective: save the results we got
    
    Inputs:
        - name, str: the name of the main model
        - nlp, spaCy.model: a spaCy model with textcat in the pipe updated
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model updated
        - scores_, dict: the dictionary of the results
    """
    
    nlp.to_disk(join(PATH_REPO, 'models', '{}_nlp'.format(name)))
    textcat.to_disk(join(PATH_REPO, 'models', '{}_textcat'.format(name)))
    pickle.dump(scores_, open(join(PATH_REPO, 'models', '{}_scores.pkl'.format(name)), 'wb'))
    
    logging.info('models saved in {}'.format(join(PATH_REPO, 'models')))
    
def restore_results(name):
    """
    Objective: restore old results
    
    Inputs:
        - name, str: the name of the main model
    Outputs:
        - nlp, spaCy.model: a spaCy model with textcat in the pipe updated
        - textcat, spaCy pipe component: the textcat pipe into the spaCy model updated
        - scores_, dict: the dictionary of the results
    """
    
    nlp = spacy.load(join(PATH_REPO, 'models', '{}_nlp'.format(name)))
    try:
        textcat = nlp.get_pipe('textcat')
    except KeyError:
        textcat = TextCategorizer(nlp.vocab)
        textcat.from_disk(join(PATH_REPO, 'models', '{}_textcat'.format(name)))
    scores_ = pickle.load(open(join(PATH_REPO, 'models', '{}_scores.pkl'.format(name)), 'rb'))
    
    logging.info('models restored from {}'.format(join(PATH_REPO, 'models')) )        
    
    return(nlp, textcat, scores_)

