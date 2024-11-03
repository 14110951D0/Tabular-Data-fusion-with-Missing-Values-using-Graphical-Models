import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class StructuralComparisonAnalyzer:
    def __init__(self, merging_results_folder, non_merging_results_folder, output_folder):
        self.merging_results = self.load_adjacency_matrices(merging_results_folder)
        self.non_merging_results = self.load_adjacency_matrices(non_merging_results_folder)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_adjacency_matrices(self, folder):
        results = {}
        for root, dirs, files in os.walk(folder):
            if 'adjacency_matrix.json' in files:
                with open(os.path.join(root, 'adjacency_matrix.json'), 'r') as f:
                    adjacency_matrix = json.load(f)
                
                folder_name = os.path.basename(root)
                parts = folder_name.split('_')
                missing_type = parts[2]
                missing_percentage = int(parts[3])
                
                results[(missing_type, missing_percentage)] = adjacency_matrix
        return results

    def calculate_structural_metrics(self, true_matrix, estimated_matrix):
        true_edges = set((i, j) for i in range(len(true_matrix)) for j in range(len(true_matrix[i])) if true_matrix[i][j])
        estimated_edges = set((i, j) for i in range(len(estimated_matrix)) for j in range(len(estimated_matrix[i])) if any(estimated_matrix[i][j].values()))
        
        tp = len(true_edges.intersection(estimated_edges))
        fp = len(estimated_edges - true_edges)
        fn = len(true_edges - estimated_edges)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    def compare_structural_metrics(self):
        results = []
        for (mt, mp), merging_matrix in self.merging_results.items():
            non_merging_matrix = self.non_merging_results.get((mt, mp))
            if non_merging_matrix:
                # Assumiamo che la vera struttura sia quella del grafo non fuso
                true_matrix = non_merging_matrix
                merging_metrics = self.calculate_structural_metrics(true_matrix, merging_matrix)
                non_merging_metrics = self.calculate_structural_metrics(true_matrix, non_merging_matrix)
                results.append({
                    'missing_type': mt,
                    'missing_percentage': mp,
                    'merging_f1': merging_metrics['f1_score'],
                    'non_merging_f1': non_merging_metrics['f1_score'],
                    'merging_precision': merging_metrics['precision'],
                    'non_merging_precision': non_merging_metrics['precision'],
                    'merging_recall': merging_metrics['recall'],
                    'non_merging_recall': non_merging_metrics['recall']
                })
        return pd.DataFrame(results)

    def plot_metric_comparison(self, metric_name):
        df = self.compare_structural_metrics()
        plt.figure(figsize=(12, 6))
        
        for mt in df['missing_type'].unique():
            mt_data = df[df['missing_type'] == mt]
            plt.plot(mt_data['missing_percentage'], mt_data[f'merging_{metric_name}'], 
                     marker='o', linestyle='-', label=f'Merging {mt}')
            plt.plot(mt_data['missing_percentage'], mt_data[f'non_merging_{metric_name}'], 
                     marker='s', linestyle='--', label=f'Non-Merging {mt}')
        
        plt.xlabel('Missing Percentage')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} Comparison')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f'{metric_name}_comparison.png'))
        plt.close()

    def plot_heatmap(self, metric_name):
        df = self.compare_structural_metrics()
        pivot_merging = df.pivot(index='missing_type', columns='missing_percentage', 
                                 values=f'merging_{metric_name}')
        pivot_non_merging = df.pivot(index='missing_type', columns='missing_percentage', 
                                     values=f'non_merging_{metric_name}')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        sns.heatmap(pivot_merging, annot=True, cmap='YlGnBu', ax=ax1)
        ax1.set_title(f'Merging {metric_name.capitalize()}')
        
        sns.heatmap(pivot_non_merging, annot=True, cmap='YlGnBu', ax=ax2)
        ax2.set_title(f'Non-Merging {metric_name.capitalize()}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f'{metric_name}_heatmap.png'))
        plt.close()

    def create_summary_table(self):
        df = self.compare_structural_metrics()
        
        summary = df.groupby(['missing_type', 'missing_percentage']).agg({
            'merging_f1': 'mean',
            'non_merging_f1': 'mean',
            'merging_precision': 'mean',
            'non_merging_precision': 'mean',
            'merging_recall': 'mean',
            'non_merging_recall': 'mean'
        }).reset_index()
        
        table = []
        for _, row in summary.iterrows():
            table.append({
                'Missing Type': row['missing_type'],
                'Missing %': row['missing_percentage'],
                'F1 (Merging)': f"{row['merging_f1']:.3f}",
                'F1 (Non-Merging)': f"{row['non_merging_f1']:.3f}",
                'Precision (Merging)': f"{row['merging_precision']:.3f}",
                'Precision (Non-Merging)': f"{row['non_merging_precision']:.3f}",
                'Recall (Merging)': f"{row['merging_recall']:.3f}",
                'Recall (Non-Merging)': f"{row['non_merging_recall']:.3f}"
            })
        
        summary_table = pd.DataFrame(table)
        summary_table.to_csv(os.path.join(self.output_folder, 'summary_table.csv'), index=False)
        
        return summary_table

    def run_analysis(self):
        for metric in ['f1_score', 'precision', 'recall']:
            self.plot_metric_comparison(metric)
            self.plot_heatmap(metric)
        
        structural_metrics = self.compare_structural_metrics()
        structural_metrics.to_csv(os.path.join(self.output_folder, 'structural_metrics.csv'), index=False)
        
        summary_table = self.create_summary_table()
        print("\nSummary Table:")
        print(summary_table.to_string(index=False))
        
        return structural_metrics, summary_table

if __name__ == "__main__":
    merging_results_folder = "experiment_results_merging_ecoli70"
    non_merging_results_folder = "experiment_results_no_merging_ecoli70"
    comparison_output_folder = "structural_comparison_results_ecoli70"

    analyzer = StructuralComparisonAnalyzer(merging_results_folder, non_merging_results_folder, comparison_output_folder)
    structural_metrics, summary_table = analyzer.run_analysis()