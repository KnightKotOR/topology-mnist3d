from gudhi import AlphaComplex
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np


def process_cloud(cloud, lifetime_thresh=1e-8, max_alpha_square=10):
    """
    Строит альфа комплекс и диаграмму персистентности для одного облака.
    """

    ac = AlphaComplex(points=cloud)

    st = ac.create_simplex_tree(
        max_alpha_square=max_alpha_square
    )

    persistence = st.persistence(homology_coeff_field=2)

    persistence = [
        (dim, pair)
        for (dim, pair) in persistence
        if np.isinf(pair[1]) or pair[1] - pair[0] > lifetime_thresh
    ]

    return st, persistence


def build_diagrams(
    point_clouds,
    lifetime_thresh=1e-8,
    max_alpha_square=10,
    n_jobs=-1
):
    """
    Параллелит построение комплексов и диаграмм для набора облаков.
    """

    trees = []
    diagrams = []

    results = Parallel(n_jobs=n_jobs)(
		delayed(process_cloud)(
        	cloud,
        	lifetime_thresh,
        	max_alpha_square
    	)
    	for cloud in tqdm(point_clouds, desc="Processing clouds")
	)

    trees, diagrams = zip(*results)

    return trees, diagrams
