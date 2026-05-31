import h5py
import numpy as np


def subsample_dataset(train_file, max_samples_per_class=10):
    """
    Сэмплирует датасет, оставляя первые max_samples_per_class облаков в каждом классе.
    """
    point_clouds = []
    labels = []

    with h5py.File(train_file, "r") as hf:
        for key in hf.keys(): # indices
            group = hf[key] # point cloud
            label = int(group.attrs["label"]) # labels

            if sum(1 for l in labels if l == label) >= max_samples_per_class:
                continue

            points = group["points"][:].astype(np.float64)

            point_clouds.append(points)
            labels.append(label)

            if len(point_clouds) >= 10 * max_samples_per_class:
                break

    labels = np.array(labels)
    return point_clouds, labels

def subsample_point_clouds(point_clouds, n_points, seed=42):
    """
    Сэмплирует облака точек, оставляя максимум n_points в облаке в соответствии с равномерным распределением.
    """
    rng = np.random.default_rng(seed=seed)

    subsampled_point_clouds = []

    for pc in point_clouds:
        if pc.shape[0] <= n_points: # если точек уже сколько нужно или меньше - оставляем как есть
            subsampled_point_clouds.append(pc)
        else:
            idx = rng.choice(pc.shape[0], n_points, replace=False)
            subsampled_point_clouds.append(pc[idx])
    subsampled_point_clouds = np.array(subsampled_point_clouds)
    return subsampled_point_clouds
