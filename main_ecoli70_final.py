import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from DataLoader import DataLoader
from DataMerger import DataMerger
from missforest import MissForestImputer2
from graph_creator import GraphCreator
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
# from Experiments.classes import *
class Redistributor:
    def __init__(self, merged_data, source_column, original_datasets):
        self.merged_data = merged_data
        self.source_column = source_column
        self.original_datasets = original_datasets

    def redistribute(self):
        imputed_datasets = {}
        
        for source, original_df in self.original_datasets.items():
            # Filtra il merged dataset per la sorgente corrente
            source_data = self.merged_data[self.merged_data[self.source_column] == source]
            
            # Rimuovi la colonna source
            source_data = source_data.drop(columns=[self.source_column])
            
            # Assicurati che gli indici corrispondano
            source_data.index = original_df.index
            
            # Combina i dati originali con quelli imputati
            # I valori imputati sostituiranno i valori mancanti nell'originale
            imputed_df = original_df.combine_first(source_data)
            
            imputed_datasets[source] = imputed_df

        return imputed_datasets 
    
def process_dataset(name, df, imputer, markov_blankets):
    imputed_df = imputer.impute(df, markov_blankets, dataset_name=name)
    pag = GraphCreator.create_pag(imputed_df)
    return name, imputed_df, pag

def extract_markov_blankets(adjacency_matrix, variables, dataset_variables):
    markov_blankets = {}
    for i, var in enumerate(variables):
        mb = set()
        for j, connection in enumerate(adjacency_matrix[i]):
            if any(connection.values()):
                mb.add(variables[j])
        for j, row in enumerate(adjacency_matrix):
            if any(row[i].values()):
                mb.add(variables[j])
        mb.discard(var)

        dataset_mb = {}
        for dataset, vars in dataset_variables.items():
            dataset_mb[dataset] = list(mb.intersection(vars))

        markov_blankets[var] = dataset_mb
    return markov_blankets

def calculate_rmse(original_data, imputed_data):
    return np.sqrt(mean_squared_error(original_data, imputed_data))

def run_imputation_pipeline(root_folder, size, missing_type, missing_percentage, max_iterations=100, convergence_threshold=0.001):
    start_time = time.time()
    
    # Load datasets
    subfolder = f"sample_size_{size}"
    missing_datasets = []
    original_datasets = []
    
    for i in range(1, 4):
        missing_file = f"data{i}_{missing_type}_{missing_percentage}.csv"
        original_file = f"data{i}_original.csv"
        
        missing_path = os.path.join(root_folder, subfolder, missing_file)
        original_path = os.path.join(root_folder, subfolder, original_file)
        
        missing_df = pd.read_csv(missing_path)
        original_df = pd.read_csv(original_path)
        
        missing_datasets.append((f"data{i}", missing_df))
        original_datasets.append((f"data{i}", original_df))
    
    # Merge datasets
    merger = DataMerger(missing_datasets)
    merged_data = merger.merge_datasets()
    
    original_merger = DataMerger(original_datasets)
    original_merged_data = original_merger.merge_datasets()
    
    # Initialize imputer
    imputer = MissForestImputer2()
    
    # Initialize variables for the loop
    iteration = 0
    rmse_history = {'merged': []}
    for i in range(1, 4):
        rmse_history[f'data{i}'] = []
    markov_blankets = None
    previous_imputation = None
    
    pbar = tqdm(total=max_iterations, desc=f"Imputation Progress (size={size}, type={missing_type}, percentage={missing_percentage})")
    
    while iteration < max_iterations:
        # Remove 'source' column before imputation
        source_column = merged_data['source']
        merged_data_without_source = merged_data.drop('source', axis=1)
        
        # Impute merged dataset
        imputed_data_without_source = imputer.impute(merged_data_without_source, markov_blankets, previous_imputation=previous_imputation)
        
        # Add 'source' column back after imputation
        imputed_merged_data = imputed_data_without_source.copy()
        imputed_merged_data['source'] = source_column
        
        # Redistribute imputed values
        redistributor = Redistributor(imputed_merged_data, 'source', dict(missing_datasets))
        imputed_datasets = redistributor.redistribute()
        
        # Impute remaining missing values in each dataset
        for name, df in imputed_datasets.items():
            imputed_datasets[name] = imputer.impute(df, markov_blankets, dataset_name=name)
        
        # Create PAGs
        pags = {}
        for name, df in imputed_datasets.items():
            pag = GraphCreator.create_pag(df)
            pags[name] = pag
        
       # Fusione dei PAGs con tutte le variabili
        all_variables = list(set(var for dataset in imputed_datasets.values() for var in dataset.columns))
        merged_edges, adjacency_matrix, variables = GraphCreator.merge_pags(list(pags.values()), [])
        
        # Estrai Markov blankets per la prossima iterazione
        dataset_variables = {name: df.columns.tolist() for name, df in imputed_datasets.items()}
        markov_blankets = extract_markov_blankets(adjacency_matrix, variables, dataset_variables)
        
        # Calculate RMSE for merged dataset
        rmse_merged = calculate_rmse(original_merged_data.drop('source', axis=1), imputed_merged_data.drop('source', axis=1))
        rmse_history['merged'].append(rmse_merged)
        
        # Calculate RMSE for individual datasets
        for i, (original_name, original_df) in enumerate(original_datasets, 1):
            imputed_df = imputed_datasets[f'data{i}']
            rmse = calculate_rmse(original_df, imputed_df)
            rmse_history[f'data{i}'].append(rmse)
        
        # Check for convergence
        if iteration > 0:
            merged_diff = abs(rmse_history['merged'][-1] - rmse_history['merged'][-2])
            individual_diffs = [abs(rmse_history[f'data{i}'][-1] - rmse_history[f'data{i}'][-2]) for i in range(1, 4)]
            if merged_diff < convergence_threshold and all(diff < convergence_threshold for diff in individual_diffs):
                print(f"\nConverged after {iteration + 1} iterations.")
                break
        
        # Update previous imputation for next iteration
        previous_imputation = imputed_merged_data.drop('source', axis=1)
        
        iteration += 1
        pbar.update(1)
    
    pbar.close()
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    results = {
        'adjacency_matrix': adjacency_matrix,
        'variables': variables,  # Aggiungi questa riga
        'rmse_history': rmse_history,
        'iterations': iteration + 1,
        'computation_time': computation_time
    }
    
    return results, imputed_datasets, imputed_merged_data

def save_results(results, imputed_datasets, imputed_merged_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Funzione di supporto per la serializzazione di numpy arrays
    def numpy_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Salva la matrice di adiacenza
    adjacency_data = {
        'matrix': results['adjacency_matrix'],
        'variables': results['variables']
    }
    with open(os.path.join(output_folder, 'adjacency_matrix.json'), 'w') as f:
        json.dump(adjacency_data, f, indent=2, default=numpy_encoder)
    
    # Save RMSE history
    with open(os.path.join(output_folder, 'rmse_history.json'), 'w') as f:
        json.dump(results['rmse_history'], f)
    
    # Save other results
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump({
            'iterations': results['iterations'],
            'computation_time': results['computation_time']
        }, f)
    
    # Save imputed datasets
    for name, df in imputed_datasets.items():
        df.to_csv(os.path.join(output_folder, f'{name}_imputed.csv'), index=False)
    
    # Save imputed merged dataset
    imputed_merged_data.to_csv(os.path.join(output_folder, 'merged_imputed.csv'), index=False)

def run_single_experiment(root_folder, output_base_folder, size, missing_type, missing_percentage):
    print(f"Running experiment: size={size}, type={missing_type}, percentage={missing_percentage}")
    
    output_folder = os.path.join(output_base_folder, f"results_{size}_{missing_type}_{missing_percentage}")
    
    results, imputed_datasets, imputed_merged_data = run_imputation_pipeline(
        root_folder, size, missing_type, missing_percentage
    )
    
    save_results(results, imputed_datasets, imputed_merged_data, output_folder)
    
    print(f"Experiment completed. Results saved in {output_folder}")

def run_all_experiments(root_folder, output_base_folder, n_jobs=-1):
    sample_sizes = [300, 600, 900]
    missing_types = ['MCAR', 'MAR', 'MNAR']
    missing_percentages = range(10, 100, 10)

    experiments = [
        (size, missing_type, missing_percentage)
        for size in sample_sizes
        for missing_type in missing_types
        for missing_percentage in missing_percentages
    ]

    Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(root_folder, output_base_folder, size, missing_type, missing_percentage)
        for size, missing_type, missing_percentage in experiments
    )

if __name__ == "__main__":
    root_folder = "generated_datasets_ecoli70"
    output_base_folder = "results_fusion_ecoli70_update"
    
    run_all_experiments(root_folder, output_base_folder, n_jobs=-1)
    
    print("All experiments completed!")

