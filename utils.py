import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from DataLoader import DataLoader
from DataMerger import DataMerger
from graph_creator import GraphCreator2
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedMissForestImputer:
    def __init__(self, max_iter=10, n_estimators=100, threshold=0.1):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.imputed_data = None

    def impute(self, data, markov_blankets=None, dataset_name=None, previous_imputation=None):
        if previous_imputation is not None:
            data_imp = previous_imputation.copy()
        else:
            data_imp = data.copy()
        
        for col in data.columns:
            if data[col].isnull().any():
                mb = markov_blankets.get(col, {}).get(dataset_name, []) if markov_blankets else []
                mb = [v for v in mb if v in data.columns]
                
                if not mb:
                    mb = [c for c in data.columns if c != col]
                
                X_train = data_imp.loc[~data[col].isnull(), mb]
                y_train = data_imp.loc[~data[col].isnull(), col]
                X_miss = data_imp.loc[data[col].isnull(), mb]
                
                if data[col].dtype in ['object', 'category']:
                    imputer = IterativeImputer(
                        estimator=RandomForestClassifier(n_estimators=self.n_estimators),
                        max_iter=self.max_iter,
                        random_state=0
                    )
                else:
                    imputer = IterativeImputer(
                        estimator=RandomForestRegressor(n_estimators=self.n_estimators),
                        max_iter=self.max_iter,
                        random_state=0
                    )
                
                try:
                    imputed_values = imputer.fit_transform(pd.concat([X_train, y_train], axis=1))[:, -1]
                    data_imp.loc[data[col].isnull(), col] = imputed_values[-sum(data[col].isnull()):]
                except Exception as e:
                    print(f"Error imputing {col}: {str(e)}. Using previous imputation or mean/mode.")
                    if previous_imputation is not None:
                        data_imp.loc[data[col].isnull(), col] = previous_imputation.loc[data[col].isnull(), col]
                    else:
                        if data[col].dtype in ['object', 'category']:
                            data_imp[col].fillna(data_imp[col].mode().iloc[0], inplace=True)
                        else:
                            data_imp[col].fillna(data_imp[col].mean(), inplace=True)
        
        return data_imp

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

def calculate_rmse(original_data, imputed_data):
    return np.sqrt(mean_squared_error(original_data, imputed_data))

def extract_markov_blankets(adjacency_matrix, variables, dataset_variables):
    markov_blankets = {}
    for i, var in enumerate(variables):
        mb = set()
        # Add parents and children
        for j, connection in enumerate(adjacency_matrix[i]):
            if any(connection.values()):
                mb.add(variables[j])
        for j, row in enumerate(adjacency_matrix):
            if any(row[i].values()):
                mb.add(variables[j])
        # Add spouses (parents of children)
        children = [j for j, connection in enumerate(adjacency_matrix[i]) if any(connection.values())]
        for child in children:
            for k, connection in enumerate(adjacency_matrix[child]):
                if any(connection.values()) and k != i:
                    mb.add(variables[k])
        mb.discard(var)

        dataset_mb = {}
        for dataset, vars in dataset_variables.items():
            dataset_mb[dataset] = list(mb.intersection(vars))

        markov_blankets[var] = dataset_mb
    return markov_blankets

def run_imputation_pipeline_with_fusion(root_folder, node_count, sample_size, missing_type, missing_percentage, max_iterations=100, convergence_threshold=0.001):
    start_time = time.time()
    
    subfolder = os.path.join(f"nodes_{node_count}", f"samples_{sample_size}")
    missing_datasets = []
    original_datasets = []
    
    for i in range(1, 4):
        missing_file = f"data{i}_{missing_type}_{missing_percentage}.csv"
        original_file = f"data{i}.csv"
        
        missing_path = os.path.join(root_folder, subfolder, missing_file)
        original_path = os.path.join(root_folder, subfolder, original_file)
        
        missing_df = pd.read_csv(missing_path)
        original_df = pd.read_csv(original_path)
        
        missing_datasets.append((f"data{i}", missing_df))
        original_datasets.append((f"data{i}", original_df))
    
    merger = DataMerger(missing_datasets)
    merged_data = merger.merge_datasets()
    
    original_merger = DataMerger(original_datasets)
    original_merged_data = original_merger.merge_datasets()
    
    imputer = IntegratedMissForestImputer()
    
    iteration = 0
    rmse_history = {'merged': []}
    for i in range(1, 4):
        rmse_history[f'data{i}'] = []
    markov_blankets = None
    previous_imputation = None
    
    pbar = tqdm(total=max_iterations, desc=f"Fusion Imputation Progress (nodes={node_count}, samples={sample_size}, type={missing_type}, percentage={missing_percentage})")
    
    while iteration < max_iterations:
        source_column = merged_data['source']
        merged_data_without_source = merged_data.drop('source', axis=1)
        
        imputed_data_without_source = imputer.impute(merged_data_without_source, markov_blankets, previous_imputation=previous_imputation)
        
        imputed_merged_data = imputed_data_without_source.copy()
        imputed_merged_data['source'] = source_column
        
        redistributor = Redistributor(imputed_merged_data, 'source', dict(missing_datasets))
        imputed_datasets = redistributor.redistribute()
        
        pags_and_vars = []
        for name, df in imputed_datasets.items():
            pag, vars = GraphCreator2.create_pag(df)
            pags_and_vars.append((pag, vars))
        
        adjacency_matrix, variables = GraphCreator2.merge_pags(pags_and_vars)
        
        dataset_variables = {name: df.columns.tolist() for name, df in imputed_datasets.items()}
        markov_blankets = extract_markov_blankets(adjacency_matrix, variables, dataset_variables)
        
        rmse_merged = calculate_rmse(original_merged_data.drop('source', axis=1), imputed_merged_data.drop('source', axis=1))
        rmse_history['merged'].append(rmse_merged)
        
        for i, (original_name, original_df) in enumerate(original_datasets, 1):
            imputed_df = imputed_datasets[f'data{i}']
            rmse = calculate_rmse(original_df, imputed_df)
            rmse_history[f'data{i}'].append(rmse)
        
        if iteration > 0:
            merged_diff = abs(rmse_history['merged'][-1] - rmse_history['merged'][-2])
            individual_diffs = [abs(rmse_history[f'data{i}'][-1] - rmse_history[f'data{i}'][-2]) for i in range(1, 4)]
            if merged_diff < convergence_threshold and all(diff < convergence_threshold for diff in individual_diffs):
                print(f"\nConverged after {iteration + 1} iterations.")
                print("\nFinal Adjacency Matrix:")
                print(json.dumps(adjacency_matrix, indent=2))
                print("\nFinal Markov Blankets:")
                print(json.dumps(markov_blankets, indent=2))
                break
        
        previous_imputation = imputed_data_without_source
        
        iteration += 1
        pbar.update(1)
    
    pbar.close()
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    results = {
        'adjacency_matrix': adjacency_matrix,
        'variables': variables,
        'rmse_history': rmse_history,
        'iterations': iteration + 1,
        'computation_time': computation_time
    }
    
    return results, imputed_datasets, imputed_merged_data

def run_imputation_pipeline_without_fusion(root_folder, node_count, sample_size, missing_type, missing_percentage, max_iterations=100, convergence_threshold=0.001):
    start_time = time.time()
    
    subfolder = os.path.join(f"nodes_{node_count}", f"samples_{sample_size}")
    missing_datasets = []
    original_datasets = []
    
    for i in range(1, 4):
        missing_file = f"data{i}_{missing_type}_{missing_percentage}.csv"
        original_file = f"data{i}.csv"
        
        missing_path = os.path.join(root_folder, subfolder, missing_file)
        original_path = os.path.join(root_folder, subfolder, original_file)
        
        missing_df = pd.read_csv(missing_path)
        original_df = pd.read_csv(original_path)
        
        missing_datasets.append((f"data{i}", missing_df))
        original_datasets.append((f"data{i}", original_df))
    
    imputer = IntegratedMissForestImputer()
    
    iteration = 0
    rmse_history = {f'data{i}': [] for i in range(1, 4)}
    markov_blankets = None
    
    pbar = tqdm(total=max_iterations, desc=f"No Fusion Imputation Progress (nodes={node_count}, samples={sample_size}, type={missing_type}, percentage={missing_percentage})")
    
    imputed_datasets = {name: df.copy() for name, df in missing_datasets}
    
    while iteration < max_iterations:
        if iteration == 0:
            # Initialize with random values for the first iteration
            for name, df in imputed_datasets.items():
                imputed_datasets[name] = df.apply(lambda x: x.fillna(x.dropna().sample(n=x.isnull().sum(), replace=True).values))
        else:
            for name, df in imputed_datasets.items():
                imputed_datasets[name] = imputer.impute(df, markov_blankets, dataset_name=name, previous_imputation=imputed_datasets[name])
        
        pags_and_vars = []
        for name, df in imputed_datasets.items():
            pag, vars = GraphCreator2.create_pag(df)
            pags_and_vars.append((pag, vars))
        
        adjacency_matrix, variables = GraphCreator2.merge_pags(pags_and_vars)
        
        dataset_variables = {name: df.columns.tolist() for name, df in imputed_datasets.items()}
        markov_blankets = extract_markov_blankets(adjacency_matrix, variables, dataset_variables)
        
        for i, (original_name, original_df) in enumerate(original_datasets, 1):
            imputed_df = imputed_datasets[f'data{i}']
            rmse = calculate_rmse(original_df, imputed_df)
            rmse_history[f'data{i}'].append(rmse)
        
        if iteration > 0:
            individual_diffs = [abs(rmse_history[f'data{i}'][-1] - rmse_history[f'data{i}'][-2]) for i in range(1, 4)]
            if all(diff < convergence_threshold for diff in individual_diffs):
                print(f"\nConverged after {iteration + 1} iterations.")
                print("\nFinal Adjacency Matrix:")
                print(json.dumps(adjacency_matrix, indent=2))
                print("\nFinal Markov Blankets:")
                print(json.dumps(markov_blankets, indent=2))
                break
        
        iteration += 1
        pbar.update(1)
    
    pbar.close()
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    results = {
        'adjacency_matrix': adjacency_matrix,
        'variables': variables,
        'rmse_history': rmse_history,
        'iterations': iteration + 1,
        'computation_time': computation_time
    }
    
    return results, imputed_datasets

def save_results(results, imputed_datasets, imputed_merged_data, output_folder, with_fusion):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(os.path.join(output_folder, 'adjacency_matrix.json'), 'w') as f:
        json.dump({
            'matrix': results['adjacency_matrix'],
            'variables': results['variables']
        }, f, indent=2)
    
    with open(os.path.join(output_folder, 'rmse_history.json'), 'w') as f:
        json.dump(results['rmse_history'], f, indent=2)
    
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump({
            'iterations': results['iterations'],
            'computation_time': results['computation_time']
        }, f, indent=2)
    
    for name, df in imputed_datasets.items():
        df.to_csv(os.path.join(output_folder, f'{name}_imputed.csv'), index=False)
    
    if with_fusion and imputed_merged_data is not None:
        imputed_merged_data.to_csv(os.path.join(output_folder, 'merged_imputed.csv'), index=False)

def run_single_experiment(root_folder, output_base_folder, node_count, sample_size, missing_type, missing_percentage, with_fusion):
    logger.info(f"Starting experiment: nodes={node_count}, samples={sample_size}, type={missing_type}, percentage={missing_percentage}, {'with' if with_fusion else 'without'} fusion")
    
    fusion_folder = os.path.join(output_base_folder, "fusion_results")
    no_fusion_folder = os.path.join(output_base_folder, "no_fusion_results")
    
    if with_fusion:
        output_folder = os.path.join(fusion_folder, f"results_nodes_{node_count}_samples_{sample_size}_{missing_type}_{missing_percentage}")
    else:
        output_folder = os.path.join(no_fusion_folder, f"results_nodes_{node_count}_samples_{sample_size}_{missing_type}_{missing_percentage}")
    
    try:
        if with_fusion:
            results, imputed_datasets, imputed_merged_data = run_imputation_pipeline_with_fusion(
                root_folder, node_count, sample_size, missing_type, missing_percentage
            )
        else:
            results, imputed_datasets = run_imputation_pipeline_without_fusion(
                root_folder, node_count, sample_size, missing_type, missing_percentage
            )
            imputed_merged_data = None
        
        save_results(results, imputed_datasets, imputed_merged_data, output_folder, with_fusion)
        
        logger.info(f"Experiment completed. Results saved in {output_folder}")
        return True
    except Exception as e:
        logger.error(f"Error in experiment: nodes={node_count}, samples={sample_size}, type={missing_type}, percentage={missing_percentage}, {'with' if with_fusion else 'without'} fusion")
        logger.error(f"Error message: {str(e)}")
        logger.error("Skipping this experiment.")
        return False

def run_all_experiments(root_folder, output_base_folder, n_jobs=-1):
    node_counts = [6, 12, 24]
    sample_sizes = [200, 400]
    missing_types = ['MCAR', 'MAR', 'MNAR']
    missing_percentages = range(10, 51, 10)  # 10%, 20%, 30%, 40%, 50%

    experiments = [
        (node_count, sample_size, missing_type, missing_percentage, with_fusion)
        for node_count in node_counts
        for sample_size in sample_sizes
        for missing_type in missing_types
        for missing_percentage in missing_percentages
        for with_fusion in [True, False]
    ]

    total_experiments = len(experiments)
    logger.info(f"Total number of experiments to run: {total_experiments}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(root_folder, output_base_folder, node_count, sample_size, missing_type, missing_percentage, with_fusion)
        for node_count, sample_size, missing_type, missing_percentage, with_fusion in tqdm(experiments, desc="Overall Progress")
    )

    successful_experiments = sum(results)
    logger.info(f"Experiments completed: {successful_experiments}/{total_experiments}")
    if successful_experiments < total_experiments:
        logger.warning(f"Failed experiments: {total_experiments - successful_experiments}")

if __name__ == "__main__":
    root_folder = "synthetic_datasets"  # Replace with the actual path to the output folder generated by the previous script
    output_base_folder = "imputation_results_synthetic_prova"
    
    # Create main output folder
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Create separate folders for fusion and no-fusion results
    os.makedirs(os.path.join(output_base_folder, "fusion_results"), exist_ok=True)
    os.makedirs(os.path.join(output_base_folder, "no_fusion_results"), exist_ok=True)

    # Setup logging to file
    log_file = os.path.join(output_base_folder, "experiment_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting all experiments")
    start_time = time.time()
    
    run_all_experiments(root_folder, output_base_folder, n_jobs=-1)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"All experiments completed! Total time: {total_time:.2f} seconds")