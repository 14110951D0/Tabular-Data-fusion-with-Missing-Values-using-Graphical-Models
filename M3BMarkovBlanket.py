import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import itertools

class M3BMarkovBlanket:
    def __init__(self, alpha=0.05, n_bins=5):
        self.alpha = alpha
        self.n_bins = n_bins

    def discretize(self, X):
        if isinstance(X, pd.Series):
            if X.empty:
                return X
            if np.issubdtype(X.dtype, np.number):
                return pd.cut(X, bins=self.n_bins, labels=False)
            return X
        elif isinstance(X, pd.DataFrame):
            return X.apply(self.discretize)
        else:
            raise ValueError("Input must be a pandas Series or DataFrame")

    def to_tuple(self, x):
        if isinstance(x, (np.ndarray, list, tuple, pd.Series)):
            return tuple(x)
        return (x,)

    def entropy(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.apply(self.to_tuple, axis=1)
        else:
            X = X.apply(self.to_tuple)
        value_counts = Counter(X)
        probs = np.array(list(value_counts.values())) / len(X)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def joint_entropy(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.apply(self.to_tuple, axis=1)
        else:
            X = X.apply(self.to_tuple)
        if isinstance(Y, pd.DataFrame):
            Y = Y.apply(self.to_tuple, axis=1)
        else:
            Y = Y.apply(self.to_tuple)
        combined = list(zip(X, Y))
        value_counts = Counter(combined)
        probs = np.array(list(value_counts.values())) / len(X)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def mutual_information(self, x, y):
        x = self.discretize(x)
        y = self.discretize(y)
        
        if x.empty or y.empty:
            return 0
        
        # Ensure x and y have the same length
        common_index = x.index.intersection(y.index)
        x = x.loc[common_index]
        y = y.loc[common_index]
        
        if len(x) == 0 or len(y) == 0:
            return 0
        
        H_X = self.entropy(x)
        H_Y = self.entropy(y)
        H_XY = self.joint_entropy(x, y)
        
        return H_X + H_Y - H_XY

    def conditional_mutual_information(self, x, y, z):
        x = self.discretize(x)
        y = self.discretize(y)
        z = self.discretize(z) if not z.empty else z
        
        if x.empty or y.empty:
            return 0
        
        # Ensure x, y, and z have the same index
        common_index = x.index.intersection(y.index).intersection(z.index if not z.empty else x.index)
        x = x.loc[common_index]
        y = y.loc[common_index]
        z = z.loc[common_index] if not z.empty else z
        
        if z.empty:
            return self.mutual_information(x, y)
        
        # Calculate conditional mutual information
        H_XZ = self.joint_entropy(x, z)
        H_YZ = self.joint_entropy(y, z)
        H_Z = self.entropy(z)
        H_XYZ = self.joint_entropy(pd.concat([x, y], axis=1), z)
        
        return H_XZ + H_YZ - H_Z - H_XYZ

    def conditional_independence_test(self, data, x, y, z):
        x_data = data.iloc[:, x]
        y_data = data.iloc[:, y]
        z_data = data.iloc[:, z] if z else pd.DataFrame()
        
        if x_data.empty or y_data.empty:
            return 1.0  # Assume independence if data is empty
        
        cmi = self.conditional_mutual_information(x_data, y_data, z_data)
        n = len(x_data)
        statistic = 2 * n * cmi
        df = (x_data.nunique() - 1) * (y_data.nunique() - 1) * \
             np.prod([z_data[col].nunique() for col in z_data.columns]) if not z_data.empty else 1
        p_value = 1 - stats.chi2.cdf(statistic, df)
        
        return p_value

    def AdjV(self, data, T):
        N = data.shape[1]
        adj_T = []
        sepset = {}
        
        for i in range(N):
            if i != T:
                if self.conditional_independence_test(data, T, i, []) > self.alpha:
                    sepset[i] = []
                else:
                    adj_T.append(i)
        
        adj_T.sort(key=lambda i: -abs(data.iloc[:, T].corr(data.iloc[:, i])))
        
        k = 1
        while k <= len(adj_T):
            for j in reversed(range(len(adj_T))):
                for S in itertools.combinations(adj_T[:j] + adj_T[j+1:], k):
                    if self.conditional_independence_test(data, T, adj_T[j], list(S)) > self.alpha:
                        sepset[adj_T[j]] = list(S)
                        adj_T.pop(j)
                        break
            k += 1
        
        return adj_T, sepset

    def discover_markov_blankets(self, data):
        N = data.shape[1]
        MMBs = {}
        self.sepset = {}
        
        for T in range(N):
            MMB_T = set()
            adj_T, self.sepset = self.AdjV(data, T)
            MMB_T.update(adj_T)
            
            for Vi in adj_T:
                for Vj in adj_T:
                    if Vi != Vj and self.conditional_independence_test(data, Vi, Vj, [T]) <= self.alpha:
                        MMB_T.add(Vj)
            
            MMBs[data.columns[T]] = list(data.columns[list(MMB_T)])
        
        return MMBs