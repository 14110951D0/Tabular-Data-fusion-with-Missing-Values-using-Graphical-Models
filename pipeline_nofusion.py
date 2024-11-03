import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from M3BMarkovBlanket import M3BMarkovBlanket
from GraphCreator import GraphCreator2
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import json

class MissForestImputer:
    def __init__(self, max_iter=10, n_estimators=100):
        self.max_iter = max_iter
        self.n_estimators = n_estimators

    def impute(self, data, markov_blankets, dataset_name):
        data_imp = data.copy()
        
        for col in data.columns:
            if data[col].isnull().any():
                mb = markov_blankets.get(col, [])
                mb = [v for v in mb if v in data.columns]
                
                if not mb:
                    mb = [c for c in data.columns if c != col]
                
                X_train = data_imp.loc[~data[col].isnull(), mb]
                y_train = data_imp.loc[~data[col].isnull(), col]
                X_miss = data_imp.loc[data[col].isnull(), mb]
                
                if data[col].dtype in ['object', 'category']:
                    clf = RandomForestClassifier(n_estimators=self.n_estimators)
                else:
                    clf = RandomForestRegressor(n_estimators=self.n_estimators)
                
                try:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_miss)
                    data_imp.loc[data[col].isnull(), col] = y_pred
                except Exception as e:
                    print(f"Error imputing {col} in {dataset_name}: {str(e)}. Using mean/mode.")
                    if data[col].dtype in ['object', 'category']:
                        data_imp[col].fillna(data_imp[col].mode().iloc[0], inplace=True)
                    else:
                        data_imp[col].fillna(data_imp[col].mean(), inplace=True)
        
        return data_imp

def preprocess_data(data):
    # Convert categorical variables to numeric
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.Categorical(data[col]).codes
    return data

def process_dataset(name, dataset, mb_finder, imputer):
    print(f"Processing {name} dataset")
    processed_data = preprocess_data(dataset)
    markov_blankets = mb_finder.discover_markov_blankets(processed_data)
    imputed = imputer.impute(processed_data, markov_blankets, name)
    pag, vars = GraphCreator2.create_pag(imputed)
    return name, imputed, pag, vars, markov_blankets

def run_imputation_pipeline(datasets, output_folder, max_iterations=1, n_jobs=-1):
    imputer = MissForestImputer()
    mb_finder = M3BMarkovBlanket()
    
    all_variables = set()
    for dataset in datasets.values():
        all_variables.update(dataset.columns)
    
    for iteration in tqdm(range(max_iterations)):
        print(f"Iteration {iteration + 1}")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_dataset)(name, dataset, mb_finder, imputer)
            for name, dataset in datasets.items()
        )
        
        imputed_datasets = {}
        pags_and_vars = []
        all_markov_blankets = {}
        
        for name, imputed, pag, vars, markov_blankets in results:
            imputed_datasets[name] = imputed
            pags_and_vars.append((pag, vars))
            all_markov_blankets[name] = markov_blankets
            
            print(f'PAG for {name}:\n', pag)
            print(f'Variables for {name}:\n', vars)
        
        # Merge PAGs
        adjacency_matrix, variables = GraphCreator2.merge_pags(pags_and_vars)
        
        variables = list(all_variables | set(variables))
    
    return imputed_datasets, all_markov_blankets, adjacency_matrix, variables

# Create output folder
output_folder = "real_world_no_fusion"
os.makedirs(output_folder, exist_ok=True)

# Load data
ppis_data = pd.read_excel('PPIS_new_excel.xlsx')
hapi_data = pd.read_csv('HAPI_processed.csv')

datasets = {
    'ppis': ppis_data,
    'hapi': hapi_data
}

# Run pipeline
imputed_datasets, markov_blankets, adjacency_matrix, variables = run_imputation_pipeline(datasets, output_folder)

# Save results
for name, imputed_data in imputed_datasets.items():
    imputed_data.to_csv(os.path.join(output_folder, f'{name}_imputed.csv'), index=False)

with open(os.path.join(output_folder, 'markov_blankets.json'), 'w') as f:
    json.dump(markov_blankets, f)

np.save(os.path.join(output_folder, 'adjacency_matrix.npy'), adjacency_matrix)
with open(os.path.join(output_folder, 'variables.txt'), 'w') as f:
    f.write('\n'.join(variables))

print(f"Pipeline completed. Results saved in the '{output_folder}' folder.")