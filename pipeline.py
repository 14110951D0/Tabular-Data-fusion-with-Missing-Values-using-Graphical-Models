import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from M3BMarkovBlanket import M3BMarkovBlanket
from GraphCreator import GraphCreator2
from tqdm import tqdm
from joblib import Parallel, delayed

class Redistributor:
    def __init__(self, merged_data, source_column, original_datasets):
        self.merged_data = merged_data
        self.source_column = source_column
        self.original_datasets = original_datasets

    def redistribute(self):
        imputed_datasets = {}
        
        for source, original_df in self.original_datasets.items():
            source_data = self.merged_data[self.merged_data[self.source_column] == source]
            source_data = source_data.drop(columns=[self.source_column])
            source_data.index = original_df.index
            imputed_df = original_df.combine_first(source_data)
            imputed_datasets[source] = imputed_df

        return imputed_datasets

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
        
        final_dataset = pd.concat(modified_datasets, ignore_index=True)
        
        return final_dataset

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
                    print(f"Error imputing {col}: {str(e)}. Using mean/mode.")
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
    print(data)
    return data



def run_imputation_pipeline(ppis_data, hapi_data, max_iterations=1, n_jobs=-1):
    # Preprocess data
    ppis_processed = preprocess_data(ppis_data)
    hapi_processed = preprocess_data(hapi_data)
    
    # Merge datasets
    merger = DataMerger([('ppis', ppis_processed), ('hapi', hapi_processed)])
    merged_data = merger.merge_datasets()
    
    # Initialize imputer and M3BMarkovBlanket
    imputer = MissForestImputer()
    mb_finder = M3BMarkovBlanket()
    
    # Find initial Markov Blankets
    markov_blankets = mb_finder.discover_markov_blankets(merged_data.drop(columns=['source']))
    
    all_variables = set(ppis_data.columns) | set(hapi_data.columns)
    
    def process_dataset(name, dataset, markov_blankets):
        imputed = imputer.impute(dataset, markov_blankets, name)
        pag, vars = GraphCreator2.create_pag(imputed)
        return name, imputed, pag, vars
    
    for iteration in tqdm(range(max_iterations)):
        print(f"Iteration {iteration + 1}")
        
        # Impute merged dataset
        imputed_merged_data = imputer.impute(merged_data.drop(columns=['source']), markov_blankets, 'merged')
        imputed_merged_data['source'] = merged_data['source']
        
        # Redistribute imputed values
        redistributor = Redistributor(imputed_merged_data, 'source', {'ppis': ppis_data, 'hapi': hapi_data})
        redistributed_datasets = redistributor.redistribute()
        
        # Parallel processing of individual datasets
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_dataset)(name, dataset, markov_blankets)
            for name, dataset in redistributed_datasets.items()
        )
        
        # Unpack results
        imputed_datasets = {}
        pags_and_vars = []
        for name, imputed, pag, vars in results:
            imputed_datasets[name] = imputed
            pags_and_vars.append((pag, vars))
        
        ppis_imputed = imputed_datasets['ppis']
        hapi_imputed = imputed_datasets['hapi']
        
        # Create merged PAG
        merged_pag, merged_vars = GraphCreator2.create_pag(imputed_merged_data.drop(columns=['source']))
        pags_and_vars.append((merged_pag, merged_vars))
        
        for p, (pag, vars) in enumerate(pags_and_vars):
            print(f'PAG {p}:\n', pag)
            print(f'Variables {p}:\n', vars)
        
        # Merge PAGs
        adjacency_matrix, variables = GraphCreator2.merge_pags(pags_and_vars)
        
        variables = list(all_variables | set(variables))
        
        # Update Markov Blankets
        markov_blankets = mb_finder.discover_markov_blankets(imputed_merged_data.drop(columns=['source']))
        
        # Update merged_data for next iteration
        merged_data = DataMerger([('ppis', ppis_imputed), ('hapi', hapi_imputed)]).merge_datasets()
    
    return ppis_imputed, hapi_imputed, imputed_merged_data, markov_blankets, adjacency_matrix, variables

# Load data
ppis_data = pd.read_excel('PPIS_new_excel.xlsx')
#print(ppis_data.dtypes)
hapi_data = pd.read_csv('HAPI_processed.csv')

# Run pipeline
ppis_imputed, hapi_imputed, merged_imputed, markov_blankets, adjacency_matrix, variables = run_imputation_pipeline(ppis_data, hapi_data)

# Save results
ppis_imputed.to_csv('ppis_imputed.csv', index=False)
hapi_imputed.to_csv('hapi_imputed.csv', index=False)
merged_imputed.to_csv('merged_imputed.csv', index=False)

import json
with open('markov_blankets.json', 'w') as f:
    json.dump(markov_blankets, f)

np.save('adjacency_matrix.npy', adjacency_matrix)
with open('variables.txt', 'w') as f:
    f.write('\n'.join(variables))

print("Pipeline completed. Results saved.")