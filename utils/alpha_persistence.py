import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from gudhi import AlphaComplex

class AlphaComplexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_alpha_square=10.0, n_jobs=-1):
        self.max_alpha_square = max_alpha_square
        self.n_jobs = n_jobs
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        def _pad_diagrams(diagrams, pad_value=0):
            max_len = max(d.shape[0] for d in diagrams)
            out = np.full((len(diagrams), max_len, 3), pad_value, dtype=np.float64)

            for i, d in enumerate(diagrams):
                n = d.shape[0]
                out[i, :n, :] = d

            return out

        def _process(cloud):
            cloud_float64 = np.asarray(cloud, dtype=np.float64)
            
            if len(cloud_float64) == 0:
                return np.empty((0, 3))
                
            ac = AlphaComplex(points=cloud_float64)
            st = ac.create_simplex_tree(max_alpha_square=self.max_alpha_square)
            persistence = st.persistence(homology_coeff_field=2)
            
            if len(persistence) == 0:
                return np.empty((0, 3))
                
            # Формат giotto-tda: [birth, death, dimension]
            diag_gtda = np.array([
                [birth, death if death != float('inf') else self.max_alpha_square, dim]
                for dim, (birth, death) in persistence
            ])

            return diag_gtda
        
        # Параллельная обработка всех облаков точек
        diagrams = Parallel(n_jobs=self.n_jobs)(
            delayed(_process)(cloud) for cloud in X
        )
        return _pad_diagrams(np.array(diagrams, dtype=object))