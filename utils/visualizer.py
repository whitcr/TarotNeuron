import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_tarot_sphere(num_points=500, embedding_dim=64, active_index=123, radius=5.0, card_name="The Fool"):
    """
    Визуализирует полную 3D-сферу трактовок карты Таро.

    :param num_points: количество трактовок (точек) на сфере
    :param embedding_dim: размерность эмбеддинга каждой трактовки
    :param active_index: индекс текущей активной трактовки
    :param radius: радиус сферы
    :param card_name: название карты
    """
    # Сферическая равномерная выборка с использованием метода Фибоначчи
    indices = torch.arange(0, num_points, dtype=torch.float) + 0.5
    phi = torch.acos(1 - 2 * indices / num_points)         # угол от вертикали
    theta = torch.pi * (1 + 5 ** 0.5) * indices           # угол по экватору

    # Вычисляем 3D координаты на сфере
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    # Вставляем в эмбеддинг
    traktovki = torch.zeros(num_points, embedding_dim)
    traktovki[:, 0] = x
    traktovki[:, 1] = y
    traktovki[:, 2] = z

    # Центр карты
    center = torch.zeros(1, 3)

    # Расстояние до активной трактовки
    distances = torch.cdist(traktovki[:, :3], traktovki[active_index, :3].unsqueeze(0))
    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
    colors = 1 - norm_distances.squeeze().numpy()  # Ближе — ярче

    # Визуализация
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Трактовки
    sc = ax.scatter(x, y, z, c=colors, cmap='plasma', s=20, label="Tractovki")

    # Центр карты
    ax.scatter(0, 0, 0, color='black', s=100, label=f"Card: {card_name}")

    # Активная трактовка
    ax.scatter(x[active_index], y[active_index], z[active_index],
               color='red', s=80, label='Active Tractovka')

    ax.set_title(f"Сфера трактовок: {card_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()
