import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import json
from joblib import Parallel, delayed
import multiprocessing

def create_random_dag(num_nodes):
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    G.add_nodes_from(nodes)
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < 0.3:  # 30% chance of edge
                G.add_edge(i, j)
    
    return G

def dag_to_adjacency_matrix(G):
    return nx.adjacency_matrix(G).todense()

def dag_to_edge_list(G):
    return list(G.edges())

def generate_dataset_from_dag(G, num_rows):
    num_nodes = G.number_of_nodes()
    data = pd.DataFrame()
    
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if not parents:
            data[f'var_{node}'] = np.random.randn(num_rows)
        else:
            parent_data = data[[f'var_{p}' for p in parents]]
            coeffs = np.random.randn(len(parents))
            data[f'var_{node}'] = parent_data.dot(coeffs) + np.random.randn(num_rows)
    
    return data

def create_subdatasets(data):
    all_vars = list(data.columns)
    common_vars = random.sample(all_vars, 4)
    
    data1_vars = common_vars + random.sample([v for v in all_vars if v not in common_vars], random.randint(1, len(all_vars)-4))
    data2_vars = common_vars + random.sample([v for v in all_vars if v not in common_vars], random.randint(1, len(all_vars)-4))
    data3_vars = common_vars + random.sample([v for v in all_vars if v not in common_vars], random.randint(1, len(all_vars)-4))
    
    return data[data1_vars], data[data2_vars], data[data3_vars]

def missing_mechanism(varnames, missingtype='MAR', rom=0.5):
    cause_dict = {}
    vars_miss = random.sample(varnames, round(len(varnames) * rom))
    vars_comp = [v for v in varnames if v not in vars_miss]
    if missingtype == 'MCAR':
        for var in vars_miss:
            cause_dict[var] = []
    elif missingtype == 'MAR':
        for var in vars_miss:
            cause_dict[var] = random.sample(vars_comp, 1)
    elif missingtype == 'MNAR':
        for var in vars_miss:
            cause_dict[var] = random.sample([v for v in vars_miss if v != var], 1)
    else:
        raise Exception('noise ' + missingtype + ' is undefined.')
    return cause_dict

def add_missing(data, cause_dict, m_min=0, m_max=0.5):
    data_missing = deepcopy(data)
    for var in cause_dict.keys():
        m = np.random.uniform(m_min, m_max)
        if len(cause_dict[var]) == 0:
            data_missing[var][np.random.uniform(size=len(data)) < m] = None
        else:
            for cause in cause_dict[var]:
                if data[cause].dtype == 'category':
                    states = list(data[cause].unique())
                    selected_state = []
                    while True:
                        state_single = random.choice(states)
                        selected_state.append(state_single)
                        states.remove(state_single)
                        if data[cause].value_counts()[selected_state].sum() > m * data.shape[0] or len(states) == 1:
                            m_h = np.random.uniform(m, m * data.shape[0] / data[cause].isin(selected_state).sum())
                            m_l = max((data.shape[0] * m - data[cause].isin(selected_state).sum() * m_h) / data[cause].isin(states).sum(), 0)
                            break
                    data_missing[var][(np.random.uniform(size=len(data)) < m_h) & (data[cause].isin(selected_state))] = None
                    data_missing[var][(np.random.uniform(size=len(data)) < m_l) & (data[cause].isin(states))] = None
                elif data[cause].dtype == 'float' or data[cause].dtype == 'int':
                    m_h = np.random.uniform(m, 1)
                    thres = data[cause].quantile(m)
                    data_missing[var][(np.random.uniform(size=len(data)) < m_h) & (data[cause] < thres)] = None
                    data_missing[var][(np.random.uniform(size=len(data)) < (1 - m_h) / (1 - m) * m) & (data[cause] >= thres)] = None
                else:
                    raise Exception('data type ' + data[cause].dtype + ' is not supported.')
    return data_missing

def generate_and_save_datasets(output_dir, node_count, sample_size):
    base_dir = os.path.join(output_dir, f"nodes_{node_count}", f"samples_{sample_size}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Create and save DAG
    G = create_random_dag(node_count)
    nx.draw(G, with_labels=True)
    plt.savefig(os.path.join(base_dir, "dag.png"))
    plt.close()
    
    # Save adjacency matrix and edge list
    adj_matrix = dag_to_adjacency_matrix(G)
    edge_list = dag_to_edge_list(G)
    np.savetxt(os.path.join(base_dir, "adjacency_matrix.csv"), adj_matrix, delimiter=",")
    with open(os.path.join(base_dir, "edge_list.json"), 'w') as f:
        json.dump(edge_list, f)
    
    # Generate and save main dataset
    data = generate_dataset_from_dag(G, sample_size)
    data.to_csv(os.path.join(base_dir, "main_dataset.csv"), index=False)
    
    # Generate and save subdatasets
    data1, data2, data3 = create_subdatasets(data)
    data1.to_csv(os.path.join(base_dir, "data1.csv"), index=False)
    data2.to_csv(os.path.join(base_dir, "data2.csv"), index=False)
    data3.to_csv(os.path.join(base_dir, "data3.csv"), index=False)
    
    # Generate datasets with missing values
    for missing_type in ['MCAR', 'MAR', 'MNAR']:
        for missing_percentage in range(10, 100, 10):
            m_min, m_max = missing_percentage / 100, missing_percentage / 100
            for i, sub_data in enumerate([data1, data2, data3], 1):
                cause_dict = missing_mechanism(list(sub_data.columns), missingtype=missing_type, rom=0.5)
                data_missing = add_missing(sub_data.copy(), cause_dict, m_min, m_max)
                data_missing.to_csv(os.path.join(base_dir, f"data{i}_{missing_type}_{missing_percentage}.csv"), index=False)

def main(output_dir):
    node_counts = range(6, 31, 6)
    sample_sizes = range(200, 1001, 200)
    
    # Determina il numero di core disponibili
    num_cores = multiprocessing.cpu_count()
    
    # Usa joblib per eseguire generate_and_save_datasets in parallelo
    Parallel(n_jobs=num_cores)(
        delayed(generate_and_save_datasets)(output_dir, node_count, sample_size)
        for node_count in node_counts
        for sample_size in sample_sizes
    )

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(output_dir)