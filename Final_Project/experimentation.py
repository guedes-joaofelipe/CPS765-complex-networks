import os
import sys
import pandas as pd
import numpy as np
import random
import logging
import multiprocessing
from random import choice
from string import ascii_uppercase
from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from caserec.utils.process_data import ReadFile
from sklearn.model_selection import train_test_split
import utils
import joblib
import json

def run_model(df_train, df_test, params):
    # Training Model
    hash_run = ''.join(choice(ascii_uppercase) for i in range(12))
    predictions_output_filepath = 'predictions_output_' + hash_run + '.dat'
    model = ItemKNN(train_file=df_train, test_file=df_test, 
        k_neighbors=params.k_neighbors, output_file=predictions_output_filepath)
    
    model.compute(verbose=False)

    # Creating evaluator with item-recommendation parameters
    evaluator = RatingPredictionEvaluation(sep = '\t', 
                                        n_rank = np.arange(1,params.rank_length+1, 1), 
                                        as_rank = True,
                                        metrics = ['PREC', 'RECALL'])

    reader = ReadFile(input_file=predictions_output_filepath)
    predictions = reader.read()
    eval_results = evaluator.evaluate(predictions['feedback'], model.test_set)   
    for evaluation in model.evaluation_results.keys():
        eval_results[evaluation] = model.evaluation_results[evaluation]
    os.remove(predictions_output_filepath)

    eval_results['hash'] = hash_run

    return eval_results


def get_overall_sparsity(df):
    return df.shape[0]/(len(df['user'].unique())*len(df['item'].unique()))

def multiprocess_realization(df_degrees, degree_user_thr, degree_item_thr, params, n_fold):

    df_filtered = utils.filter_dataset(df_degrees, degree_user=degree_user_thr, 
            degree_item=degree_item_thr, strategy=params.strategy)
    df_filtered = df_filtered[['from', 'to', 'rating']]
    df_filtered.columns = ['user', 'item', 'feedback_value']

    df_train, df_test = train_test_split(df_filtered, test_size=params.test_size, random_state=n_fold)
    logging.info("Train size: {} \nTest size: {}".format(df_train.shape[0], df_test.shape[0]))

    # try:
    eval_results = run_model(df_train, df_test, params)
    eval_results['degree_user_thr'] = degree_user_thr
    eval_results['degree_item_thr'] = degree_item_thr    

    eval_results['os'] = get_overall_sparsity(df_filtered)
    eval_results['os_train'] = get_overall_sparsity(df_train)
    eval_results['os_test'] = get_overall_sparsity(df_test)
    # except:
    #     print('Error Running Model for user thr {} [{} users] and item thr {} [{} items]\n')
        

    filepath = os.path.join('.', 'Experiments', params.hash, 
        'fold_'+str(n_fold), 'evaluation_results', eval_results['hash'] + '.json')

    with open(filepath, 'w') as json_file:
        json.dump(eval_results, json_file, indent=10, separators=(',', ': '))

    

def run_experimentation(df_degrees, df_exp,  n_fold, params):

    experiment_output_folder = os.path.join('.', 'Experiments', params.hash, 
        'fold_'+str(n_fold))        
    utils.create_folder(experiment_output_folder)

    utils.create_folder(os.path.join(experiment_output_folder, 'evaluation_results'), if_exists='remove')
    utils.set_logger(os.path.join(experiment_output_folder, 'logger.log'))       

    logging.info('======================')
    logging.info('   START EXPERIMENT   ')
    logging.info('======================\n')
    logging.info('params: {}'.format(str(params)))
    
    step_size = 10
    degree_thr = np.array([float(x)/100.0 for x in np.logspace(2, -2, step_size, dtype=float)])
    n_users_max = len(df_degrees['from'].unique())
    n_items_max = len(df_degrees['to'].unique())
    logging.info('max number of users: {}'.format(str(n_users_max)))
    logging.info('max number of items: {}\n'.format(str(n_items_max)))

    processes = list()
    for index, realization in df_exp.iterrows():
        logging.info('Running realization {}/{}'.format(index+1, df_exp.shape[0]))
        degree_user_thr = realization['degree_centrality_users']
        degree_item_thr = realization['degree_centrality_items']    
        p = multiprocessing.Process(target=multiprocess_realization, args=(df_degrees, degree_user_thr, degree_item_thr, params, n_fold))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    logging.info('\n')
    logging.info('End of experiment for folder {} (hash)'.format(n_fold))    
    logging.info('============================================')

if __name__ == '__main__':
    
    data_path = './Variables'
    config_file = './config_file.json'

    params = utils.Params(config_file)
    df_degrees = pd.read_csv(os.path.join(data_path, 'df_degrees.csv'), sep=';')
    df_exp = pd.read_csv(os.path.join(data_path, 'df_exp.csv'), sep=';')

    print ("Starting experimentation ", params.hash)

    for n_fold in np.arange(params.n_folds):        
        run_experimentation(df_degrees, df_exp, n_fold, params)