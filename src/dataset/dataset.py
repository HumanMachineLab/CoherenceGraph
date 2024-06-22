import sys
import os
import pandas
import json
import config

sys.path.append("../../")

default_data_path = os.path.join("..", "data")


class RawData:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.split = None

    def get_data(self, split="train"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/{self.dataset_type}/wikisection_en_{self.dataset_type}_{split}.json"
        data = None
        with open(data_path) as f:
            data = json.load(f)

        assert data is not None, "data could not be imported"

        return data
    
    def get_qmsum_data(self, split="train", dataset="Academic"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/qmsum/{dataset}/{split}"
        data = []

        for filename in os.listdir(data_path):
            f = os.path.join(data_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f) as f:
                    temp_data = json.load(f)
                    assert data is not None, f"data: {filename} could not be imported"
                    data.append(temp_data)
        
        assert len(data) > 0, "data could not be imported"

        return data
    
    def get_superseg_data(self, split="train"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/superseg/segmentation_file_{split}.json"
        data = None
        with open(data_path) as f:
            data = json.load(f)

        assert data is not None, "data could not be imported"

        return data
    
    
    def get_wiki_data(self):
        data_path = f"{config.root_path}/raw_data/wiki727k"
        data = []

        for filename in os.listdir(data_path):
            f = os.path.join(data_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f, mode="r", encoding="utf-8") as f:
                    for line in f:
                        data.append(line)
        
        assert len(data) > 0, "data could not be imported"
    
    
    def get_choi_data(self):
        keys = ["3-5", "3-11", "6-8", "9-11"]
        data = {}

        for key in keys:
            data_path = f"{config.root_path}/raw_data/choi/2/{key}"
            data[key] = []
            for filename in os.listdir(data_path):
                f = os.path.join(data_path, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    with open(f, mode="r", encoding="utf-8") as f:
                        for line in f:
                            data[key].append(line)
        
            assert len(data[key]) > 0, "data could not be imported"

        return data
    
    
    def get_manifesto_data(self):
        data_path = f"{config.root_path}/raw_data/manifesto"
        data = []

        for filename in os.listdir(data_path):
            f = os.path.join(data_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f, mode="r", encoding="utf-8") as f:
                    for line in f:
                        data.append(line)
        
        assert len(data) > 0, "data could not be imported"

        return data
    
    def get_ami_data(self):
        data_path = f"{config.root_path}/raw_data/ami"
        data = []

        for filename in os.listdir(data_path):
            if filename == ".DS_Store": continue
            f = os.path.join(data_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                file = open(f)
                new_data = json.load(file)
                data.extend(new_data)
                file.close()
        
        assert len(data) > 0, "data could not be imported"

        return data
    
    def get_icsi_data(self):
        data_path = f"{config.root_path}/raw_data/icsi"
        icsi_data = []

        for filename in os.listdir(data_path):
            f = os.path.join(data_path, filename)
            if filename == ".DS_Store": continue
            # checking if it is a file
            if os.path.isfile(f):
                with open(f, 'rb') as f:
                    data = f.read()
                    data_str = data.decode("utf-8", errors='ignore')
                    new_data = json.loads(data_str)
                    icsi_data.extend(new_data)
        
        assert len(icsi_data) > 0, "data could not be imported"

        return icsi_data


class DatasetMixin:
    # save dataset and retrieve dataset information.
    # load datasets based on parameters.
    @staticmethod
    def get_datasets():
        df = pandas.read_csv(os.path.join(default_data_path, "datasets.csv"))
        print(df)

    @staticmethod
    def save_dataset():
        pass
