from textcat_utils import*
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train or test a model')
    
    parser.add_argument('--is_predict', metavar='t', type=bool, default=False,
                   help='whether you want to predict and save predictions')
    
    parser.add_argument('--is_evaluate', metavar='t', type=bool, default=False,
                   help='whether you want to evaluate the model')
    
    parser.add_argument('--name_model', metavar='n', type=str, default = None, 
                   help='name of already existing model')
    
    parser.add_argument('--cv', metavar='c', type=int, default = 5, 
                   help='# of cv for training')
    
    parser.add_argument('--model', metavar='m', type=str, default='RF',
                   help='name fo the model, RF or GC')
   
    parser.add_argument('--X', metavar='X', type=str, default='Text',
                   help='The name of the feature set')
   
    parser.add_argument('--y', metavar='y', type=str, default='Cat',
                   help='The name of the label set')
    
    parser.add_argument('--min_trees', metavar='mt', type=int, default=50,
                        help='The minimum number of trees for the CV')
    
    parser.add_argument('--max_trees', metavar='Mt', type=int, default=150,
                        help='The minimum number of trees for the CV')
    
    parser.add_argument('--step_trees', metavar='st', type=int, default=10,
                        help='The step for the number of trees for CV')
    
    parser.add_argument('--min_depth', metavar='md', type=int, default=2,
                        help='The minimum depth of the tree for the CV')
    
    parser.add_argument('--max_depth', metavar='Md', type=int, default=4,
                        help='The minimum depth of the tree for the CV')
    
    parser.add_argument('--step_depth', metavar='sd', type=int, default=1,
                        help='The step for the depth of the tree for CV')
   
    args = parser.parse_args()
    
    
    if args.is_evaluate:
        X, y = load_data('{}_test'.format(args.X), '{}_test'.format(args.y))
        X = preprocess_tf_hub(X)
        clf = restore_model(args.name_model, args.model)
        scores, cm = test_sklearn_model(X, y, clf)
    elif args.is_predict:
        X_t, y = load_data('{}'.format(args.X))
        X = preprocess_tf_hub(X_t)
        clf = restore_model(args.name_model, args.model)
        y_pred = clf.predict(X)
        save_prediction('{}'.format(args.X), X_t, y_pred)
    else:
        X, y = load_data('{}_train'.format(args.X), '{}_train'.format(args.y))
        X = preprocess_tf_hub(X)
        params_grid = {"n_estimators": range(args.min_trees, args.max_trees, args.step_trees),
                       "max_depth": range(args.min_depth, args.max_depth, args.step_depth)}
        clf = train_sklearn_model(X, y, model=args.model, CV=args.cv, **params_grid)
        save_model(args.name_model, args.model, clf)
        