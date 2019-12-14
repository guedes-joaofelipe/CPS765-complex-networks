import json
import hashlib
import os

def create_hash(config_file, limit=5):
    with open(config_file) as f:
        json_file = json.load(f)

    hash = hashlib.md5(json.dumps(json_file, sort_keys=True).encode()).hexdigest()

    return hash[:limit]

def create_folder(complete_path):
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    return 0

class Params():
    """
    Class to load model hyperparameters from a json file.
    """
    def __init__(self, json_path):
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
    def dict(self):
        return self.__dict__

if __name__ == '__main__':
    config_file = './config_file.json'
    hash = create_hash(config_file)
    print ('Hash: ', hash)

    params = Params('./config_file.json')
    print (params)