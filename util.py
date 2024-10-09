import torch
import numpy as np
from typing import List
import time
import threading

import open3d as o3d

def draw_pc(pc, path,labels=None):
    open3d_pc = o3d.geometry.PointCloud()
    pc = pc.cpu().numpy()
    open3d_pc.points = o3d.utility.Vector3dVector(pc[:, :3])
    open3d_pc.normals = o3d.utility.Vector3dVector(pc[:, 3:])
    
    if labels is not None:
        labels = labels.cpu().numpy()
        label2color = {lb: np.random.rand(3) for lb in np.unique(labels)}
        colors = [label2color[lb] for lb in labels]
        open3d_pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), open3d_pc)
    return open3d_pc
    

def gen_grid(n=10):
    index = torch.arange(0, n ** 3)
    z = index % n
    xy = index // n
    y = xy % n
    x = xy // n
    pts = torch.stack([x, y, z], dim=1).float()
    pts = pts / n
    pts -= 0.5
    pts *= 2
    return pts


def orient_center(pred):
    cent = pred[:, :3].mean(dim=0)
    ref = pred[:, :3] - cent
    flip_mask = (ref * pred[:, 3:]).sum(dim=-1) < 0
    pred[flip_mask, 3:] *= -1
    return pred

def export_pc(pc, dest):
    txt = '\n'.join(map(lambda x: ' '.join(map(lambda y: str(y.item()), x)), pc.transpose(0, 1)))
    txt.strip()

    with open(dest, 'w+') as file:
        file.write(txt)

def xyz2tensor(txt, append_normals=True):
    pts = []
    for line in txt.split('\n'):
        line = line.strip()
        spt = line.split(' ')
        if 'nan' in line:
            continue
        if len(spt) == 6:
            pts.append(torch.tensor([float(x) for x in spt]))
        if len(spt) == 3:
            t = [float(x) for x in spt]
            if append_normals:
                t += [0.0 for _ in range(3)]
            pts.append(torch.tensor(t))

    rtn = torch.stack(pts, dim=0)
    return rtn

def npxyz2tensor(np_pc, append_normals=True):
    if np_pc.shape[1] == 3:
        if append_normals:
            np_pc = np.concatenate([np_pc, np.zeros((np_pc.shape[0], 3))], axis=1)
        return torch.tensor(np_pc)
    else:
        return torch.tensor(np_pc)


# 计算每个点在grid网络中的索引，而不是判断是否在某个grid中
def _lzd_divide_pc(pc_in: torch.Tensor, n_part: int, ranges=(-1.5, 1.5),
                   min_patch=0):
    x_idx_map = torch.linspace(ranges[0], ranges[1], n_part + 1).to(pc_in.device)
    y_idx_map = torch.linspace(ranges[0], ranges[1], n_part + 1).to(pc_in.device)
    z_idx_map = torch.linspace(ranges[0], ranges[1], n_part + 1).to(pc_in.device)
    x_idx = torch.searchsorted(x_idx_map, pc_in[:, 0], right=True) - 1
    y_idx = torch.searchsorted(y_idx_map, pc_in[:, 1], right=True) - 1
    z_idx = torch.searchsorted(z_idx_map, pc_in[:, 2], right=True) - 1
    index = x_idx + y_idx * n_part + z_idx * n_part * n_part
    unique_idx = torch.unique(index)
    unique_idx = np.array(unique_idx.cpu())
    index_map = {unique_idx[i]: i for i in range(len(unique_idx))}
    index = index.cpu().numpy()
    indices = [[] for _ in range(len(unique_idx))]
    for i in range(len(pc_in)):
        indices[index_map[index[i]]].append(i)
    # 转为tensor
    indices = [[torch.tensor(i)] for i in indices]
    point_ijk = torch.stack([x_idx, y_idx, z_idx], dim=1)
    patch_ijk = [[point_ijk[patch[0][0]]] for patch in (indices)] 
    return indices, patch_ijk
   
def _divide_pc(pc_in: torch.Tensor, n_part: int, ranges=(-1.5, 1.5),
              min_patch=0):
    '''
    divide a pc into voxel parts
    Args:
        pc_in: input pc (N X (3/6))
        n_part: number of parts in each axis i.e. total num_parts ** 3 parts
        ranges: range of the bounding box the pc is in
        min_patch: join patches with less than min_path points

    Returns: 
    indices : List[torch.Tensor] a list of indices corresponding to each part
    ijk : List[torch.Tensor] a list of grid indices corresponding to each element in indices
    '''
    def mask_to_index(mask, n):
        return torch.arange(n).to(mask.device)[mask]

    def bounds(t):
        l = edge_len * t + ranges[0]
        return l, l + edge_len

    pc = pc_in[:, :3]
    num_points = pc.shape[0]
    indices = []
    ijk = []
    edge_len = (ranges[1] - ranges[0]) / (n_part)
    for i in range(n_part + 1):
        x1, x2 = bounds(i)
        x_mask = (x1 < pc[:, 0]) * (pc[:, 0] <= x2)
        for j in range(n_part + 1):
            y1, y2 = bounds(j)
            y_mask = (y1 < pc[:, 1]) * (pc[:, 1] <= y2)
            for k in range(n_part + 1):
                z1, z2 = bounds(k)
                z_mask = (z1 < pc[:, 2]) * (pc[:, 2] <= z2)

                total_mask = x_mask * y_mask * z_mask
                if total_mask.long().sum() > 0:
                    indices.append([mask_to_index(total_mask, num_points)])
                    ijk.append([torch.tensor([i, j, k])])
    return indices, ijk
    
def divide_pc_to_graph(pc_in: torch.Tensor, n_part: int, ranges=(-1.5, 1.5),
                min_patch=0, edge_calculator=None,point_estimator=None):        
        # 并行执行point_estimator
        def thread_func(i):
            if point_estimator is not None:
                point_estimator(pc_in[indices[i]])
        MyTimer = util.timer_factory()
        
        # with MyTimer('divide pc into grid'):
        #     indices, ijk = _divide_pc(pc_in, n_part, ranges, min_patch)                 
        
        with MyTimer('lzd_divide_pc'):
            indices, ijk = _lzd_divide_pc(pc_in, n_part, ranges, min_patch)
            
        with MyTimer('merge nodes'):
            indices, ijk = merge_nodes(pc_in[:, :3], indices, ijk, min_patch)    
        
        def if_neibor(i, j):
            idxi = ijk[i][0]
            idxj = ijk[j][0]
            if (idxi - idxj).abs().sum() == 1:
                return 1   
            
        with MyTimer('point estimator'):  
            threads = []
            for i in range(len(indices)):
                t = threading.Thread(target=thread_func, args=(i,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        
        
        with MyTimer('edge calculator'):
            G = BidGraph()
            G.V = [i for i in range(len(indices))]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    # 判断i,j是否相邻
                    if not if_neibor(i, j):
                        continue
                    if edge_calculator is not None:    
                        w, invw = edge_calculator(pc_in[indices[i]], pc_in[indices[j]])
                    else:
                        assert False
                    G.E.append(BiEdge(i, j, w, invw))
        return G, indices
              


def divide_pc(pc_in: torch.Tensor, n_part: int, ranges=(-1.5, 1.5),
              min_patch=0) -> (List[torch.Tensor], List[torch.Tensor]):
    indices, ijk = _divide_pc(pc_in, n_part, ranges, min_patch)
    return indices

import open3d as o3d
from graph import *
def draw_topology(G,pc,patches,nodelabel = [],edgelabel = [],path = None):
    '''
    用open3d画出图的拓扑结构
    其中 每个节点,在center处用get_sphere画一个球
    每个边,根据其起点和终点的位置,用get_arrow
    '''
    if len(nodelabel) == 0:
        nodelabel = np.zeros(len(G.V))
    if len(edgelabel) == 0: 
        edgelabel = np.zeros(len(G.E))
    
    def get_V_center(V):
        p = pc[patches[V]]
        center = p.mean(dim=0)
        return center.cpu().numpy()[0:3]
    
    mesh = ([], [])
    colors = []
    unque_label = list(set(nodelabel).union(set(edgelabel)))
    label2color = {}
    for i in range(len(unque_label)):
        label2color[unque_label[i]] = np.random.rand(3)
    
    
    for i in range(len(G.V)):
        center = get_V_center(i)
        sp = get_sphere(center)
        add_topology(mesh,get_sphere(center))
        colors += [label2color[nodelabel[i]] for _ in range(len(sp[0]))]
    for i in range(len(G.E)):
        start = get_V_center(G.E[i].u)
        end = get_V_center(G.E[i].v)
        arrow = get_arrow(start,end)
        add_topology(mesh,arrow)
        colors += [label2color[edgelabel[i]] for _ in range(len(arrow[0]))]    
    o3dmesh = o3d.geometry.TriangleMesh()
    o3dmesh.vertices = o3d.utility.Vector3dVector(mesh[0])
    o3dmesh.triangles = o3d.utility.Vector3iVector(mesh[1])
    o3dmesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([o3dmesh])
    if path is not None:
        o3d.io.write_triangle_mesh(str(path), o3dmesh)
    return o3dmesh


def merge_nodes(pts, indices, ijk, min_patch):
    def find_dij(i, ijk1, ijks):
        min_ijks = -1
        for other_index, ijk2 in enumerate(ijks):
            if other_index != i:
                for i_sub in range(len(ijk1)):
                    for j_sub in range(len(ijk2)):
                        dij = (ijk1[i_sub] - ijk2[j_sub])  # look for neighbors
                        if (dij.abs() <= 1).all():
                            min_ijks = other_index
                            break


        return min_ijks

    remaining_small_patches = True
    count = 0
    max_recursive_merges = 10
    while remaining_small_patches and count < max_recursive_merges:
        remaining_small_patches = False
        count += 1
        for i in range(len(ijk)):
            if len(indices[i]) > 0 and len(indices[i][0]) < min_patch:
                if len(ijk[i]) > 0:
                    min_j = find_dij(i, ijk[i], ijk)
                    if min_j != -1:
                        indices[min_j][0] = torch.cat([indices[min_j][0], indices[i][0]])
                        for t in range(len(ijk[i])):
                            ijk[min_j].append(ijk[i][t])
                        indices[i] = []
                        ijk[i] = []
                        if len(indices[min_j][0]) < min_patch:
                            remaining_small_patches = True

    if count == max_recursive_merges:
        print('recursive merge failed to merge some patches')

    new_indices = []
    new_ijk = []
    for i in range(len(ijk)):
        if len(ijk[i]) > 0 and len(indices[i][0]) >= min_patch:
            new_indices.append(torch.cat(indices[i]))
            new_ijk.append(ijk[i])

    return new_indices, new_ijk


def pca_eigen_values(x: torch.Tensor):
    temp = x[:, :3] - x.mean(dim=0)[None, :3]
    cov = (temp.transpose(0, 1) @ temp) / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)
    n = v[:, 0]
    return e[0:1], n


def rotate_to_principle_components(x: torch.Tensor, scale=True):
    temp = x[:, :3] - x.mean(dim=0)[None, :3]
    cov = temp.transpose(0, 1) @ temp / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)

    # rotate xyz
    rotated = x[:, :3]@v
    if scale:
        # scale to unit var on for the larger eigen value
        rotated = rotated / torch.sqrt(e[2])

    # if x contains normals rotate the normals as well
    if x.shape[1] == 6:
        rotated = torch.cat([rotated, x[:, 3:]@v], dim=-1)
    return rotated


def estimate_normals_torch(inputpc, max_nn):
    from torch_cluster import knn_graph
    knn = knn_graph(inputpc[:, :3], max_nn, loop=False)
    knn = knn.view(2, inputpc.shape[0], max_nn)[0]
    x = inputpc[knn][:, :, :3]
    temp = x[:, :, :3] - x.mean(dim=1)[:, None, :3]
    cov = temp.transpose(1, 2) @ temp / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)
    n = v[:, :, 0]
    return torch.cat([inputpc[:, :3], n], dim=-1)



import pymeshlab as ml
def meshlab_estimate_normal(inputpc,smoothiter=0):
    xyz = inputpc[:, :3].cpu().numpy()
    m = ml.Mesh(xyz)
    ms = ml.MeshSet()
    ms.add_mesh(m)
    ms.compute_normal_for_point_clouds(smoothiter = 0)    
    normals = ms.current_mesh().vertex_normal_matrix()
    inputpc = torch.cat((inputpc[:, :3], torch.Tensor(normals).to(inputpc.device)), dim=1)
    return inputpc


'''
    Estimate normals for a point cloud using open3d
    inputpc: torch.Tensor, point cloud (N X 3)
    max_nn: int, number of nearest neighbors to consider
    keep_orientation: bool, if True, the orientation of the normals is kept (建议设为False,防止使用到ground truth) 
'''
def estimate_normals(inputpc, max_nn=30, keep_orientation=False):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        xyz = np.array(inputpc[:, :3].cpu())
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=max_nn))
        normals = np.array(pcd.normals)
        inputpc_unoriented = torch.cat((inputpc[:, :3], torch.Tensor(normals).to(inputpc.device)), dim=1)
        if keep_orientation:
            flip = (inputpc[:, 3:] * inputpc_unoriented[:, 3:]).sum(dim=-1) < 0
            inputpc_unoriented[flip, 3:] *= -1
    except ModuleNotFoundError:
        inputpc_unoriented = estimate_normals_torch(inputpc, max_nn)

    return inputpc_unoriented


class Transform:
    def __init__(self, pc: torch.Tensor, ttype='reg'):
        if ttype == 'reg':
            self.center = pc[:, :3].mean(dim=0)
            self.scale = (pc[:, :3].max(dim=0)[0] - pc[:, :3].min(dim=0)[0]).max()
        elif ttype == 'bb':
            self.center = pc[:, :3].mean(dim=0)
            pc_tag = pc[:, :3] - self.center
            d = pc[:, :3].sum(dim=-1)
            a, b = d.argmin(), d.argmax()
            line = pc_tag[b] - pc_tag[a]
            self.scale = line.norm()
            mid_points = (pc_tag[a] + pc_tag[b]) / 2
            self.center += mid_points

    def apply(self, pc: torch.Tensor) -> torch.Tensor:
        pc = pc.clone()
        pc[:, :3] -= self.center[None, :]
        pc[:, :3] = pc[:, :3] / self.scale
        return pc

    def inverse(self, pc: torch.Tensor) -> torch.Tensor:
        pc = pc.clone()
        pc[:, :3] = pc[:, :3] * self.scale
        pc[:, :3] += self.center[None, :]
        return pc

    @staticmethod
    def trans(pc: torch.Tensor, ttype='reg'):
        T = Transform(pc, ttype=ttype)
        return T.apply(pc), T


def timer_factory():
    class MyTimer(object):
        total_count = 0

        def __init__(self, msg='', count=True):
            self.msg = msg
            self.count = count

        def __enter__(self):
            self.start = time.perf_counter()
            if self.msg:
                print(f'started: {self.msg}')
            return self

        def __exit__(self, typ, value, traceback):
            self.duration = time.perf_counter() - self.start
            if self.count:
                MyTimer.total_count += self.duration
            if self.msg:
                print(f'finished: {self.msg}. duration: {MyTimer.convert_to_time_format(self.duration)}')

        @staticmethod
        def print_total_time():
            print('\n ----- \n')
            print(f'total time: {MyTimer.convert_to_time_format(MyTimer.total_count)}')

        @staticmethod
        def convert_to_time_format(sec):
            sec = round(sec, 2)
            if sec < 60:
                return f'{sec} [sec]'

            minutes = int(sec / 60)
            remaining_seconds = sec - (minutes * 60)
            remaining_seconds = round(remaining_seconds, 2)
            return f'{minutes}:{remaining_seconds} [min:sec]'

    return MyTimer


