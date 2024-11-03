import pandas as pd

class DataMerger:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def merge_datasets(self):
        modified_datasets = []
        common_columns = set.intersection(*(set(df.columns) for _, df in self.datasets))
        
        for name, df in self.datasets:
            df_filtered = df[list(common_columns)].copy()
            df_filtered['source'] = name
            modified_datasets.append(df_filtered)
        
        final_dataset = pd.concaxt(modified_datasets, ignore_index=True)
        
        return final_dataset
    
# data1 = pd.DataFrame({
# 'name': ['Alice', 'Bob'],
# 'age': [25, 30],
# 'city': ['New York', 'Los Angeles']
# }, index=[0, 1])

# data1.name = 'data1'  # Assegnazione del nome al dataset

# data2 = pd.DataFrame({
#     'name': ['Chris', 'David'],
#     'age': [35, 45],
#     'city': ['Chicago', 'Miami'],
#     'alchol': [1,0]
# }, index=[0, 1])
# data2.name = 'data2'

# data = [data1, data2]
# # Creazione dell'oggetto DatasetMerger
# merger = DatasetMerger(data)
# # Fusione dei dataset
# result = merger.merge_datasets()
# print(f'DATA1: {data1}')
# print(f'DATA2: {data2}')
# print(f'DATA_MERGED: {result}')