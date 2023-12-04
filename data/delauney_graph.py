import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


# 生成随机点
points = [
    [0,0],
    [1,1],
    [2,-3],
    [-5,-2],
    [-4,6],
    [-9,-1],
    [10,0]
]
points = np.array(points)

# 计算Voronoi图
vor = Voronoi(points)

# 计算绘图范围
min_x, min_y = -10, -10
max_x, max_y = 10,10
padding = 0.1  # 额外的填充空间
x_range = max_x - min_x
y_range = max_y - min_y
max_range = max(x_range, y_range)
x_lim = (min_x - padding * max_range, max_x + padding * max_range)
y_lim = (min_y - padding * max_range, max_y + padding * max_range)

# 绘制Voronoi图
fig, ax = plt.subplots(figsize=(6, 6))  # 设置画布大小为正方形
voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.savefig('voronoi.png')
print(f'voronoi.png saved')