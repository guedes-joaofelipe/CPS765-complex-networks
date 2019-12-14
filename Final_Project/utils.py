import json
import hashlib
import os
import logging
from sklearn.model_selection import train_test_split
import shutil

# def create_hash(config_file, limit=5):
#     with open(config_file) as f:
#         json_file = json.load(f)

#     hash = hashlib.md5(json.dumps(json_file, sort_keys=True).encode()).hexdigest()

#     return hash[:limit]

def create_folder(complete_path, if_exists=None):
    if os.path.exists(complete_path) and if_exists == 'remove':
        remove_folder(complete_path)

    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    return 0

def remove_folder(complete_path):
    if os.path.exists(complete_path):
        shutil.rmtree(complete_path, ignore_errors=True)

    return 0

class Params():
    """
    Class to load model hyperparameters from a json file.
    """
    def __init__(self, json_path):
        self.config_file = json_path
        self.update(json_path)

    def __str__(self):
        return str(self.dict)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def hash(self, limit=5):
        with open(self.config_file) as f:
            json_file = json.load(f)
        file_hash = hashlib.md5(json.dumps(json_file, sort_keys=True).encode()).hexdigest()

        return file_hash[:limit]

    @property
    def dict(self):
        return self.__dict__

def filter_dataset(df_degrees, degree_user=None, degree_item=None, split_size=.3, strategy='upper_degree_centrality'):
    degree_user = 1 if degree_user == None else degree_user
    degree_item = 1 if degree_item == None else degree_item
    
    if strategy == 'upper_degree_centrality':       
        cond_user = df_degrees['degree_centrality_user'] > degree_user
        cond_item = df_degrees['degree_centrality_item'] > degree_item            
        df_result = df_degrees[cond_user & cond_item]        
    
    elif strategy == 'lower_degree_centrality':
        cond_user = df_degrees['degree_centrality_user'] <= degree_user
        cond_item = df_degrees['degree_centrality_item'] <= degree_item            
        df_result = df_degrees[cond_user & cond_item]    
    
    elif strategy == 'random':
        df_result, _ = train_test_split(df_degrees, test_size=split_size)
        df_result.reset_index(drop=True, inplace=True)
        return df_result
        
    return df_result

def set_logger(log_path, if_exists='remove'):
    """
    Example:
    -------
        logging.info("Start training...")
    """
    
    # As file at filePath is deleted now, so we should check if file exists or not not before deleting them
    if os.path.exists(log_path) and if_exists == 'remove':
        os.remove(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


if __name__ == '__main__':
    config_file = './config_file.json'
    params = Params(config_file)   
    print ('Hash: ', params.hash)
    
    