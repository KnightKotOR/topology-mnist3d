import numpy as np

def add_gaussian_noise(clouds: np.ndarray, sigma: float = 0.001, seed=42):
    """
    Добавляет гауссовский шум к облакам точек.
    """
    np.random.seed(seed)
    noisy_clouds = []
    for cloud in clouds:
        noise = np.random.normal(0, sigma, cloud.shape)
        noisy_clouds.append(cloud + noise)
    return noisy_clouds
