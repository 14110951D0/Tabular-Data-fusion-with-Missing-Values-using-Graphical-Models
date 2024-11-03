import os
import pandas as pd

class DataLoader:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def load_datasets(self, size, missing_type, missing_percentage):
        if missing_type == 'original':
            return self.load_original_datasets(size)
        else:
            return self.load_missing_datasets(size, missing_type, missing_percentage)

    def load_original_datasets(self, size):
        folder_path = os.path.join(self.root_folder, f"sample_size_{size}")
        datasets = []
        for i in range(1, 4):
            file_path = os.path.join(folder_path, f"data{i}_original.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                datasets.append((f"data{i}", df))
        return datasets

    def load_missing_datasets(self, size, missing_type, missing_percentage):
        folder_path = os.path.join(self.root_folder, f"sample_size_{size}")
        datasets = []
        for i in range(1, 4):
            file_name = f"data{i}_{missing_type}_{missing_percentage}.csv"
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                datasets.append((file_name, df))
        return datasets