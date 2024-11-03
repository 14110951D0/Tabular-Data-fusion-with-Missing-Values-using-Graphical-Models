class Redistributor:
    def __init__(self, merged_data, source_column, original_datasets):
        self.merged_data = merged_data
        self.source_column = source_column
        self.original_datasets = original_datasets

    def redistribute(self):
        imputed_datasets = {}
        for source, group in self.merged_data.groupby(self.source_column):
            original_df = self.original_datasets[source]
            imputed_df = group.drop(columns=[self.source_column])
            imputed_df.index = original_df.index
            imputed_datasets[source] = imputed_df
        return imputed_datasets