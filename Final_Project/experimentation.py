import os
import sys
import pandas as pd
import numpy as np
import random
import logging
import multiprocessing
import networkx as nx
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
    return 1-df.shape[0]/(len(df['user'].unique())*len(df['item'].unique()))

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
    eval_results['n_users'] = df_filtered['user'].unique().shape[0]
    eval_results['n_users_train'] = df_train['user'].unique().shape[0]
    eval_results['n_items'] = df_filtered['item'].unique().shape[0]
    eval_results['n_items_train'] = df_train['item'].unique().shape[0]
    eval_results['n_evals'] = df_filtered.shape[0]
    
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
    
    # step_size = 10
    # degree_thr = np.array([float(x)/100.0 for x in np.logspace(2, -2, step_size, dtype=float)])
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
    logging.info('End of experiment for folder {} (hash {})'.format(n_fold, params.hash))
    logging.info('============================================')

def create_degrees_dataset(params):

    print ('[*] Creating Degrees Dataset')
    # Loading Dataset
    dataset_file = os.path.join('..', 'Data', params.dataset, 'ratings.dat')
    df = pd.read_csv(dataset_file, header=None, sep=r'::', engine="python")
    df.columns = ["from", "to", "rating", "timestamp"]
    df['from'] = df['from'].apply(lambda x: 'user_' + str(x))
    df['to'] = df['to'].apply(lambda x: 'item_' + str(x))
    
    # Creating Graph
    G = nx.DiGraph()
    for user_id in df["from"].unique():
        G.add_node(user_id, node_type="user")
    for item_id in df["to"].unique():
        G.add_node(item_id, node_type="item")
    G.add_weighted_edges_from(df[["from", "to", "rating"]].values)

    # Getting Stats
    df_stats = pd.DataFrame(nx.degree(G))
    df_stats.columns = ['node', 'degree']
    df_stats['degree_centrality'] = nx.degree_centrality(G).values()

    # Associating stats with original dataset
    df_degrees = pd.merge(df, df_stats, how='inner', left_on='from', right_on='node').drop(['node'], axis=1)
    df_degrees = pd.merge(df_degrees, df_stats, how='inner', left_on='to', right_on='node', 
                   suffixes=['_user', '_item']).drop(['node'], axis=1)
    df_degrees.to_csv(os.path.join('.', 'Variables', 'df_degrees.csv'), sep=';', index=None)
    print ('[+] Degrees Dataset Created')

    return df_degrees

def create_experiment_dataset(df_degrees, params):
    """Creates a dataset containing the thresholds to be used in the 
    the experiments. This is a way of saving processing time by not 
    having repeated threshold scenarios to be processed."""
    
    print ('[*] Creating Experiment Dataset')
    if params.strategy == 'random':
        degree_thr = np.array([float(x)/100.0 for x in np.arange(10, 101, params.step_size)])
    else:    
        degree_thr = np.array([float(x)/100.0 for x in np.logspace(-2, 2, params.step_size, dtype=float)])

    n_users_thr, n_items_thr = [], []
    user_degree_thr, item_degree_thr = [], []
    n_users_max = len(df_degrees['from'].unique())
    n_items_max = len(df_degrees['to'].unique())
    
    for threshold in degree_thr:    
        if params.strategy == 'upper_degree_centrality':
            df_temp = df_degrees[df_degrees['degree_centrality_user'] > threshold]
        elif params.strategy == 'lower_degree_centrality':
            df_temp = df_degrees[df_degrees['degree_centrality_user'] < threshold]        
        elif params.strategy == 'random':
            df_temp = utils.filter_dataset(df_degrees, degree_user=threshold, degree_item=1, strategy='random')

        n_users = len(df_temp['from'].unique())    
        n_users_thr.append(n_users)
        user_degree_thr.append(threshold)
        if n_users == 0 and params.strategy == 'upper_degree_centrality':
            break
        if n_users == n_users_max and params.strategy == 'lower_degree_centrality':
            break  
            
    for threshold in degree_thr:    
        if params.strategy == 'upper_degree_centrality':
            df_temp = df_degrees[df_degrees['degree_centrality_item'] > threshold]
        elif params.strategy == 'lower_degree_centrality':
            df_temp = df_degrees[df_degrees['degree_centrality_item'] < threshold]        
        elif params.strategy == 'random':
            df_temp = utils.filter_dataset(df_degrees, degree_user=1, degree_item=threshold, strategy='random')

        n_items = len(df_temp['to'].unique())    
        n_items_thr.append(n_items)
        item_degree_thr.append(threshold)
        if n_items == 0 and params.strategy == 'upper_degree_centrality':
            break
        if n_items == n_items_max and params.strategy == 'lower_degree_centrality':
            break  

    arr = []
    for u, u_thr in enumerate(user_degree_thr):
        for i, i_thr in enumerate(item_degree_thr):
            arr.append([u_thr, n_users_thr[u], i_thr, n_items_thr[i]])
    df_exp = pd.DataFrame(arr, columns=['degree_centrality_users', 'n_users', 'degree_centrality_items', 'n_items'])
    cond1 = df_exp['n_users'] > 5
    cond2 = df_exp['n_items'] > 5
    df_exp = df_exp[cond1 & cond2]
    df_exp.to_csv(os.path.join('.', 'Variables', 'df_exp.csv'), sep=';', index=None)
    print ('[+] Experiment Dataset Created')

    return df_exp

if __name__ == '__main__':
    
    data_path = './Variables'
    config_file = './config_file.json'

    params = utils.Params(config_file)
    df_degrees = create_degrees_dataset(params)
    df_exp = create_experiment_dataset(df_degrees, params)
    # df_degrees = pd.read_csv(os.path.join(data_path, 'df_degrees.csv'), sep=';')
    # df_exp = pd.read_csv(os.path.join(data_path, 'df_exp.csv'), sep=';')

    print ("Starting experimentation ", params.hash)
    experiment_output_folder = os.path.join('.', 'Experiments', params.hash)
    utils.create_folder(experiment_output_folder)
    with open(os.path.join(experiment_output_folder, 'config_file.json'), 'w') as fp: 
        # copy config file into output directory
        json.dump(params.dict, fp)

    for n_fold in np.arange(params.n_folds):        
        run_experimentation(df_degrees, df_exp, n_fold, params)