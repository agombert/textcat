from textcat_utils import*
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train or test a model')
    
    parser.add_argument('--is_predict', metavar='t', type=bool, default=False,
                   help='whether you want to predict or not')
    
    parser.add_argument('--name_model', metavar='n', type=str, default = None, 
                   help='name of already existing model')
    
    parser.add_argument('--cats', metavar='C', type=str, nargs='+',
                   help='categories for your text classifications')
   
    parser.add_argument('--X', metavar='X', type=str, default='Text',
                   help='The name of the feature set')
   
    parser.add_argument('--y', metavar='y', type=str, default='Cat',
                   help='The name of the label set')
   
    parser.add_argument('--bs_m', metavar='m', type=int, default=4,
                   help='The minimum batch size for training')
   
    parser.add_argument('--bs_M', metavar='M', type=str, default='Cat',
                   help='The maximum batch size for training')
   
    parser.add_argument('--step', metavar='s', type=float, default=1.001,
                   help='The step to go from min batchsize to max batchsize')
   
    parser.add_argument('--epoch', metavar='e', type=int, default=10,
                   help='Number of epoch for the training')
   
    parser.add_argument('--drop', metavar='d', type=float, default=0.5,
                   help='Drop out to apply the model')
   
    parser.add_argument('--verbose', metavar='v', type=int, default=1,
                   help='To have verbosity or not')
   
    args = parser.parse_args()
    
    try:
        nlp, textcat, scores_ = restore_results(args.name_model)
    except:
        nlp, textcat, scores_ = load_model_labels(*args.cats)
    
    if args.is_predict:
        X, y = load_data('{}_test'.format(args.X), '{}_test'.format(args.y))
        evaluate_test(X, y, textcat, nlp)
    else:
        X, y = load_data('{}_train'.format(args.X), '{}_train'.format(args.y))
        train_data, dev_texts, dev_cats = preprocess_before_training(textcat, X, y, r=0.8)
        nlp, textcat, scores_ = training(nlp, textcat, train_data, dev_texts, dev_cats,
                                         bs_min=4, bs_max=32, step=1.001,
                                         n_iter=args.epoch, drop=args.drop, verbose=args.verbose,
                                         scores_ = scores_)
        save_results(args.name_model, nlp, textcat, scores_)
        