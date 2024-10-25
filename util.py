import torch
import numpy as np
from typing import List
import time
import threading

import open3d as o3d

def draw_pc(pc, path,labels=None):
    open3d_pc = o3d.geometry.PointCloud()
    # pc = pc.cpu().numpy()
    if type(pc) == torch.Tensor:
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


'''
计算每个点在grid网络中的索引，而不是判断是否在某个grid中
return: 
    indices: List[torch.Tensor] a list of indices in pc_in corresponding to each part
    patch_ijk: List[torch.Tensor] a list of grid indices corresponding to each element in indices
'''
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


'''
point_estimator: function, 用于估计每个patch的点法向量
将点云划分为图,每个节点为一个patch
return: 
    G: BidGraph, 图
    indices: List[torch.Tensor] a list of indices, indices[i]表示G的第i个节点的点在pc_in中的索引
'''
def divide_pc_to_graph(pc_in: torch.Tensor, n_part: int, ranges=(-1.5, 1.5),
                min_patch=0, edge_calculator=None,point_estimator=None):        
        MyTimer = util.timer_factory()
        
        # with MyTimer('divide pc into grid'):
        #     indices, ijk = _divide_pc(pc_in, n_part, ranges, min_patch)                 
        
        with MyTimer('lzd_divide_pc'):
            indices, ijk = _lzd_divide_pc(pc_in, n_part, ranges, min_patch)
            
        with MyTimer('merge nodes'):
            # indices, ijk= merge_nodes(pc_in, indices, ijk, min_patch)
            indices, ijk, ijk_source = lzd_merge_nodes(pc_in, indices, ijk, min_patch)
            # ijk = [i[0] for i in ijk]
            # indices = [i[0] for i in indices]
            # ijk_source = [[ijk[i]] for i in range(len(indices))]
        
        
        def if_neibor(ijk_source1, ijk_source2):
            for i in range(len(ijk_source1)):
                for j in range(len(ijk_source2)):
                    if (ijk_source1[i] - ijk_source2[j]).abs().sum() == 1:
                        return True
            return False  
            
        with MyTimer('point estimator'):  
             # 并行执行point_estimator
            def thread_func(i):
                if point_estimator is not None:
                    pc_in[indices[i]] = point_estimator(pc_in[indices[i]])
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
                    if not if_neibor(ijk_source[i], ijk_source[j]):
                        continue
                    if edge_calculator is not None:    
                        w, invw = edge_calculator(pc_in[indices[i]], pc_in[indices[j]])
                    else:
                        assert False
                    G.E.append(BiEdge(i, j, w, invw))
        return G, indices


def if_pc_bbox_intersect(pc1, pc2):
    assert type(pc1) == np.ndarray and type(pc2) == np.ndarray
    assert pc1.shape[1] == 3 and pc2.shape[1] == 3
    min1 = pc1.min(axis=0)
    max1 = pc1.max(axis=0)
    min2 = pc2.min(axis=0)
    max2 = pc2.max(axis=0)
    return (min1 <= max2).all() and (max1 >= min2).all()

'''
判断两个点云是否存在距离小于threshold的点对
'''
def if_pc_neibor(pc1,pc2,threshold):
    pc1 = pc1[:, :3]
    pc2 = pc2[:, :3]
    assert (type(pc1) == np.ndarray and type(pc2) == np.ndarray) or (type(pc1) == torch.Tensor and type(pc2) == torch.Tensor)
    # 判断是否存在某个轴上的距离大于threshold 如果有则不相邻
   
    if type(pc1) == np.ndarray:
        if (pc1.min(axis=0) - pc2.max(axis=0) > threshold).any() or (pc1.max(axis=0) - pc2.min(axis=0) < -threshold).any():
            return False
    
        for i in range(len(pc1)):
            dist = np.linalg.norm(pc1[i] - pc2, axis=1)
            if np.min(dist) < threshold:
                return True
            
    else:
        if (pc1.min(dim=0).values - pc2.max(dim=0).values > threshold).any() or (pc1.max(dim=0).values - pc2.min(dim=0).values < -threshold).any():
            return False
        for i in range(len(pc1)):
            dist  = torch.norm(pc1[i] - pc2, dim=1)
            if torch.min(dist) < threshold:
                return True
    return False


'''
计算k近邻的最远距离的中位数
'''
def avg_min_dist(pc: np.ndarray, k: int):
    # 建立kd树
    from sklearn.neighbors import KDTree
    tree = KDTree(pc[:, :3])
    if len(pc) < k + 1:
        k = len(pc) - 1
    assert k > 0
    dist, _ = tree.query(pc[:, :3], k=k + 1)
    return np.median(dist[:, -1])

'''
k_neighbors: int, number of neighbors to consider
mininum_rate: float, 当叶子节点的点数小于mininum_rate * len(xyz)时，停止划分


'''
def divide_pc_by_ncut(pc_in: torch.Tensor, k_neighbors, mininum_rate,
                      edge_calculator=None,point_estimator=None
                      ):
    # import toolboox.pointcloud_segmentation.cluster as cluster
    # import toolboox.pointcloud_segmentation.bitree_cluster as bc
    # import open3d as o3d
    from toolbox.pointcloud_segmentation.socket_server_para import bitree_cluster_plus
    xyz = pc_in.cpu().numpy()
    xyz = np.array(xyz[:, :3])
    conf = {'k_neighbors': k_neighbors, 'mininum_rate': mininum_rate}
    MyTimer = util.timer_factory()
    with MyTimer('bitree_cluster_plus'):
        labels = bitree_cluster_plus(conf, xyz)    
        indices = []
        for i in range(len(set(labels))):
            indices.append(torch.tensor(np.where(labels == i)[0]))
        
    
    
    with MyTimer('point estimator'):
        # 并行执行point_estimator
        def thread_func(i):
            if point_estimator is not None:
                pc_in[indices[i]] = point_estimator(pc_in[indices[i]])
        MyTimer = util.timer_factory()
        threads = []
        for i in range(len(indices)):
            t = threading.Thread(target=thread_func, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            
    
    with MyTimer('build_graph'): 
        G_mutex = threading.Lock()

        def cal_edge_ij_thread(i,j):
            if edge_calculator is not None:
                if if_pc_neibor(pc_in[indices[i]], pc_in[indices[j]], threshold):
                    w,invw = edge_calculator(pc_in[indices[i]], pc_in[indices[j]])
                    G_mutex.acquire()
                    G.E.append(BiEdge(i, j, w, invw))
                    G_mutex.release()
            return 
        
        threshold = avg_min_dist(xyz, k_neighbors)
        G = BidGraph()
        G.V = [i for i in range(len(indices))]
        threads = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                t = threading.Thread(target=cal_edge_ij_thread, args=(i,j))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()
                
    return G, indices
    
           

'''
return: indices: List[torch.Tensor] a list of indices corresponding to each part
eg: indices[0] = [1,2,3,4,5,6,7,8,9,10], indices[1] = [11,12,13,14,15,16,17,18,19,20], 
for 20 points and n_part = 2
'''
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
    unique_edge_label = np.unique(np.array(edgelabel))
    unique_node_label = np.unique(np.array(nodelabel))
    
    node_label2color = {unique_node_label[i]: np.random.rand(3) for i in range(len(unique_node_label))}
    edge_label2color = {unique_edge_label[i]: np.random.rand(3) for i in range(len(unique_edge_label))}
    
    if len(unique_node_label) == 2:
        node_label2color = {unique_node_label[0]: np.array([1,0,0]), unique_node_label[1]: np.array([0,1,0])}
    if len(unique_edge_label) == 2:
        edge_label2color = {unique_edge_label[0]: np.array([1,0,0]), unique_edge_label[1]: np.array([0,1,0])}
    
    
    for i in range(len(G.V)):
        center = get_V_center(i)
        sp = get_sphere(center)
        add_topology(mesh,get_sphere(center))
        colors += [node_label2color[nodelabel[i]] for _ in range(len(sp[0]))]
    for i in range(len(G.E)):
        start = get_V_center(G.E[i].u)
        end = get_V_center(G.E[i].v)
        arrow = get_arrow(start,end)
        add_topology(mesh,arrow)
        colors += [edge_label2color[edgelabel[i]] for _ in range(len(arrow[0]))]    
    o3dmesh = o3d.geometry.TriangleMesh()
    o3dmesh.vertices = o3d.utility.Vector3dVector(mesh[0])
    o3dmesh.triangles = o3d.utility.Vector3iVector(mesh[1])
    o3dmesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([o3dmesh])
    if path is not None:
        o3d.io.write_triangle_mesh(str(path), o3dmesh)
    return o3dmesh

'''
indices: List[torch.Tensor] a list of indices corresponding to each part
ijk: List[torch.Tensor] a list of grid indices corresponding to each element in indices
min_patch: join patches with less than min_path points
after merge, each patch has at least min_patch points

return:
    new_indices: List[torch.Tensor] a list of indices corresponding to each part
    new_ijk: List[torch.Tensor] a list of grid indices corresponding to each element in indices
    ijk_source : List[List[torch.Tensor]] 每个patch的来源
'''
def lzd_merge_nodes(pts, indices, ijk, min_patch):
    ijk = [i[0] for i in ijk] #不知道dipole作者为什么要用list包裹ijk
    indices = [i[0] for i in indices]
    ori_patch_cnt = len(indices) 
    def if_neighbor(s1,s2):
        ijk_source1 = [ijk[i] for i in s1]
        ijk_source2 = [ijk[i] for i in s2]
        if len(ijk_source1) == 0 or len(ijk_source2) == 0:
            return False 
        for i in range(len(ijk_source1)):
            for j in range(len(ijk_source2)):
                if (ijk_source1[i] - ijk_source2[j]).abs().sum() == 1:
                    return True
        return False
    
    ijk_source_idx = [[i] for i in range(ori_patch_cnt)] # 每个patch分配一个列表，记录其来源。最开始时，每个patch只有一个来源；合并后，每个patch可能有多个来源
    pt_count = np.array([len(i) for i in indices])
    
    for i,count in enumerate(pt_count):
        if count > min_patch:
            continue
        neighbor = np.array([j for j in range(ori_patch_cnt) if i!=j and if_neighbor(ijk_source_idx[i], ijk_source_idx[j])])
        if len(neighbor) == 0:
            continue
        smallest_neighbor = neighbor[np.argmin(pt_count[neighbor])]
        ijk_source_idx[smallest_neighbor] += ijk_source_idx[i]
        ijk_source_idx[i] = []
        pt_count[neighbor[0]] += count
        pt_count[i] = 0
    new_indices = []
    new_ijk = []
    ijk_source = []
    for i in range(ori_patch_cnt):
        if ijk_source_idx[i] == []:
            continue
        new_indices.append(torch.cat([indices[j] for j in ijk_source_idx[i]]))
        new_ijk.append(ijk[i])
        ijk_source.append([ijk[j] for j in ijk_source_idx[i]])
    return new_indices, new_ijk, ijk_source
    
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
        print('open3d not found, using torch_cluster instead')
        inputpc_unoriented = estimate_normals_torch(inputpc, max_nn)

    return inputpc_unoriented

def doing_nothing(inputpc):
    # warning: this function does nothing
    if inputpc.shape[1] != 6:
        print(inputpc.shape)
        assert False
    print('doing nothing\n')
    return inputpc

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

'''
加载点云数据并转换为tensor
'''
import open3d as o3d
def load_and_trans_tensor(path, device=torch.device('cuda')):
    pc = o3d.io.read_point_cloud(path)
    data = np.asarray(pc.points)
    if pc.has_normals():
        normals = np.asarray(pc.normals)
        data = torch.tensor(np.concatenate([data, normals], axis=1), dtype=torch.float32, device=device)
    else :
        data = torch.tensor(data, dtype=torch.float32, device=device)
    data, trans = Transform.trans(data)
    return trans, data

def cal_nd_loss(pc1, pc2):
    n1 = pc1[:, 3:]
    n2 = pc2[:, 3:]
    # 计算平均角度差
    cos = (n1 * n2).sum(dim=1)
    cos = cos.clamp(-1, 1)
    angle = torch.acos(cos)
    angle = angle.mean()
    angle = angle * 180 / 3.1415926
    loss = min(angle.item(), 180 - angle.item())
    return loss

def cal_90_count(pc1,pc2):
    n1 = pc1[:, 3:]
    n2 = pc2[:, 3:]
    # 计算平均角度差
    cos = (n1 * n2).sum(dim=1)
    cos = cos.clamp(-1, 1)
    angle = torch.acos(cos)
    angle = angle * 180 / 3.1415926
    count = (angle < 90).sum().item()
    count = min(count, len(angle) - count)
    return count

def cal_loss(pc1, pc2):
    assert pc1.shape[1] == pc2.shape[1]
    loss = cal_nd_loss(pc1, pc2)
    count = cal_90_count(pc1, pc2)
    return "loss: {:.2f}, 90_count: {}/{}".format(loss, count, len(pc1))