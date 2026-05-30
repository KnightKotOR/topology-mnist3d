import pandas as pd
import h5py

from pathlib import Path


def compute_statistics(h5_path):
    """
    Вычисляет статистику по облакам точек.
    Возвращает pandas DataFrame с глобальной статистикой и статистикой по каждой цифре.
    """
    h5_path = Path(h5_path)
    
    stats = {
        "num_points": [],
        "range_x": [],
        "range_y": [],
        "range_z": [],
        "label": []
    }
    
    with h5py.File(h5_path, "r") as hf:
        for key in hf.keys():
            points = hf[key]["points"][:]      # (N, 3)
            label = int(hf[key].attrs["label"])
            
            # Bounding box
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            ranges = maxs - mins
            
            stats["num_points"].append(len(points))
            stats["range_x"].append(ranges[0])
            stats["range_y"].append(ranges[1])
            stats["range_z"].append(ranges[2])
            stats["label"].append(label)
    
    df = pd.DataFrame(stats)
    
    class_stats = df.groupby('label').agg({
        'num_points': ['count', 'mean', 'std', 'min', 'max'],
        'range_x': ['mean', 'std'],
        'range_y': ['mean', 'std'],
        'range_z': ['mean', 'std']
    }).round(4)
    
    class_stats = class_stats.rename(columns={'count': 'num_clouds'}, level=1)
    
    global_stats = df.describe().T[['mean', 'std', 'min', 'max']].round(4)

    
    return class_stats, global_stats
