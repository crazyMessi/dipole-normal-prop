import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes, mesh_surface_area
# from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go


def plot_F(F, x=np.linspace(-5, 5, 100), y=np.linspace(-5, 5, 100), z=np.linspace(-5, 5, 100)):
    """
    绘制：
    1. x=0 的二维横截面
    2. y=0 的二维横截面
    3. z=0 的二维横截面
    4. 三维等值面
    """

    # 计算场
    X, Y, Z = np.meshgrid(x, y, z)
    F_values = F(X, Y, Z)

    # 绘制 x=0 的二维横截面
    F_x_0 = F_values[int(len(x) / 2), :, :]  # x=0 时的 F 值
    X2, Y2 = np.meshgrid(y, z)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].contour(X2, Y2, F_x_0,cmap='viridis')
    axes[0, 0].set_title("F at x=0")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("z")

    # 绘制 y=0 的二维横截面
    F_y_0 = F_values[:, int(len(y) / 2), :]  # y=0 时的 F 值
    X2, Z2 = np.meshgrid(x, z)
    axes[0, 1].contour(X2, Z2, F_y_0,cmap='viridis')
    axes[0, 1].set_title("F at y=0")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("z")

    # 绘制 z=0 的二维横截面
    F_z_0 = F_values[:, :, int(len(z) / 2)]  # z=0 时的 F 值
    X2, Y2 = np.meshgrid(x, y)
    axes[1, 0].contour(X2, Y2, F_z_0,cmap='viridis')
    axes[1, 0].set_title("F at z=0")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    # 绘制三维等值面
    fig3d = go.Figure(data=go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=F_values.flatten(),
        isomin=0.05,  # 设置最小值
        isomax=1.0,  # 设置最大值
        opacity=0.5,  # 透明度
        surface_count=10,  # 设置等值面的数量
        colorscale='Viridis'
    ))

    fig3d.update_layout(
        title="Interactive Isosurface of F",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    # 显示二维和三维图形
    plt.tight_layout()
    plt.show()
    fig3d.show()


def draw_F(F, x=np.linspace(-5, 5, 100), y=np.linspace(-5, 5, 100), z=np.linspace(-5, 5, 100), levels=10, filename="isosurface_point_cloud.ply",verbose=False):
    """
    使用 marching cubes 提取等值面，并保存为 .ply 文件
    :param F: 计算函数，返回电场或标量值
    :param x: 网格中的 x 坐标
    :param y: 网格中的 y 坐标
    :param z: 网格中的 z 坐标
    :param levels: 绘制等值面数量
    :param filename: 保存点云的文件名
    """
    # 计算场
    X, Y, Z = np.meshgrid(x, y, z)
    F_values = F(X, Y, Z)

    # 创建一个列表来保存所有的点
    all_points = []
    # all_faces = []
    all_colors = []

    # normalize F_values
    F_values = F_values / np.max(F_values)
    
    level2color = plt.cm.viridis(np.linspace(0, 1, levels))
    
    # 计算等值面
    for level in np.linspace(np.min(F_values), np.max(F_values), levels):        
        # 使用 marching cubes 提取等值面
        try:
            verts, faces, _, _ = marching_cubes(F_values,level)
        except:
            print(f"等值面提取失败，level={level}")
            
            continue
        
        # 转换到实际坐标系
        verts = verts / np.array([len(x), len(y), len(z)]) * np.array([x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]]) + np.array([x[0], y[0], z[0]])
        
        # 添加到列表
        all_points.extend(verts)
        colors = np.tile(level2color[int((level-np.min(F_values))/(np.max(F_values)-np.min(F_values))*(levels-1))], (len(verts), 1))
        all_colors.extend(colors)
        
        # all_faces.extend(faces)

    # 转换为 numpy 数组
    all_points = np.array(all_points)
    all_colors = np.array(all_colors)
    all_colors = all_colors[:, :3] # 去掉 alpha 通道
    # all_faces = np.array(all_faces)

    # 使用 open3d 创建点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 保存点云
    o3d.io.write_point_cloud(filename, point_cloud)

    if not verbose:
        return

    # 可视化点云和网格
    o3d.visualization.draw_geometries([point_cloud])


