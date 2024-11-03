import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import seaborn as sns
from causallearn.search.ConstraintBased.FCI import fci 
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils

    
class GraphCreator2:
    @staticmethod
    def create_pag(data):
        if 'source' in data.columns:
            data = data.drop(columns=['source'])
        
        G, edges = fci(data.values, fisherz, 0.05, verbose=False)
        
        edge_list = []
        for edge in edges:
            edge_str = str(edge)
            parts = edge_str.split()
            if len(parts) == 3:
                var1, relation, var2 = parts
                edge_list.append(f"{data.columns[int(var1[1:])-1]} {relation} {data.columns[int(var2[1:])-1]}")
        
        return edge_list, list(data.columns)

    @staticmethod
    def merge_pags(pags_and_vars, weighted_edges=[]):
        all_variables = set()
        for edges, vars in pags_and_vars:
            all_variables.update(vars)
        all_variables = sorted(list(all_variables))
        #print("ALL VARIABLES:\n", all_variables)
        n = len(all_variables)
        
        adjacency_matrix = [[{'o-o': 0, '-->': 0, '<->': 0, '--': 0, 'o->': 0} for _ in range(n)] for _ in range(n)]
        
        var_to_index = {var: idx for idx, var in enumerate(all_variables)}
        
        for edges, _ in pags_and_vars:
            for edge in edges:
                #print("EDGE:\n", edge)
                parts = edge.split()
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed edge: {edge}")
                    continue
                
                # Trova l'indice del simbolo di relazione
                relation_index = next((i for i, part in enumerate(parts) if part in ['o-o', '-->', '<->', '--', 'o->']), -1)
                if relation_index == -1:
                    print(f"Warning: No valid relation found in edge: {edge}")
                    continue
                
                var1 = ' '.join(parts[:relation_index])
                relation = parts[relation_index]
                var2 = ' '.join(parts[relation_index+1:])
                
                if var1 in var_to_index and var2 in var_to_index:
                    i, j = var_to_index[var1], var_to_index[var2]
                    
                    adjacency_matrix[i][j][relation] += 1/len(pags_and_vars)
                    if relation == 'o-o':
                        adjacency_matrix[j][i][relation] += 1/len(pags_and_vars)
                    elif relation == '<->':
                        adjacency_matrix[j][i][relation] += 1/len(pags_and_vars)
                else:
                    print(f"Warning: Unknown variable in edge: {edge}")
        
        for edge in weighted_edges:
            print("EDGE:\n",edge)
            parts = edge.split()
            relation_index = next((i for i, part in enumerate(parts) if part in ['o-o', '-->', '<->', '--', 'o->']), -1)
            if relation_index == -1:
                print(f"Warning: No valid relation found in weighted edge: {edge}")
                continue
            
            var1 = ' '.join(parts[:relation_index])
            relation = parts[relation_index]
            var2 = ' '.join(parts[relation_index+1:])
            
            if var1 in var_to_index and var2 in var_to_index:
                i, j = var_to_index[var1], var_to_index[var2]
                
                adjacency_matrix[i][j][relation] += 1
                if relation == 'o-o':
                    adjacency_matrix[j][i][relation] += 1
                elif relation == '<->':
                    adjacency_matrix[j][i][relation] += 1
            else:
                print(f"Warning: Unknown variable in weighted edge: {edge}")
        
        return adjacency_matrix, all_variables