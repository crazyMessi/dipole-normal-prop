import torch
import numpy as np
import util
import pdb


def measure_mean_potential(pc: torch.Tensor):
    grid = util.gen_grid().to(pc.device)
    return potential(pc, grid).mean()


def potential(sources, means, eps=1e-5, recursive=True, max_pts=15000):
    """
    Calculate dipole potential
    Args:
        sources: position and dipole moments for the fields sources
        means: positions to calculate potential at

    Returns:
        torch.Tensor meansX3 field at measurement positions
    """

    if recursive:
        def break_by_means():
            mid = int(means.shape[0] / 2)
            return torch.cat([potential(sources, means[:mid], eps, recursive, max_pts),
                              potential(sources, means[mid:], eps, recursive, max_pts)], dim=0)

        def break_by_sources():
            mid = int(sources.shape[0] / 2)
            return potential(sources[:mid], means, eps, recursive, max_pts) \
                   + potential(sources[mid:], means, eps, recursive, max_pts)

        if sources.shape[0] > max_pts and means.shape[0] > max_pts:
            if sources.shape[0] > means.shape[0]:
                return break_by_sources()
            else:
                return break_by_means()

        if sources.shape[0] > max_pts:
            return break_by_sources()

        if means.shape[0] > max_pts:
            return break_by_means()

    p = sources[:, 3:]
    R = sources[:, None, :3] - means[None, :, :3]

    phi = (p[:, None, :] * R).sum(dim=-1)
    phi = phi / (R.norm(dim=-1) ** 3)[:, :]
    phi_total = phi.sum(dim=0)

    phi_total[phi_total.isinf()] = 0
    phi_total[phi_total.isnan()] = 0
    return phi_total


'''
lzd修改:考虑了距离为0的点
'''
def field_grad(sources, means, eps=1e-5, recursive=True, max_pts=15000):
    """
    Calculate dipole field i.e. potential gradient
    Args:
        sources: position and dipole moments for the fields sources
        means: positions to calculate gradient at

    Returns:
        torch.Tensor meansX3 field at measurement positions
    """

    # recursive calculation for large number of points（if the number of points is too large, the calculation will be divided into two parts）
    if recursive:
        def break_by_means():
            mid = int(means.shape[0] / 2)
            return torch.cat([field_grad(sources, means[:mid], eps, recursive, max_pts),
                              field_grad(sources, means[mid:], eps, recursive, max_pts)], dim=0)

        def break_by_sources():
            mid = int(sources.shape[0] / 2)
            return field_grad(sources[:mid], means, eps, recursive, max_pts) \
                   + field_grad(sources[mid:], means, eps, recursive, max_pts)

        if sources.shape[0] > max_pts and means.shape[0] > max_pts:
            if sources.shape[0] > means.shape[0]:
                return break_by_sources()
            else:
                return break_by_means()

        if sources.shape[0] > max_pts:
            return break_by_sources()

        if means.shape[0] > max_pts:
            return break_by_means()

    p = sources[:, 3:]
    R = sources[:, None, :3] - means[None, :, :3]
    # 排除距离为0的点产生的电场
    zero_mask = R.norm(dim=-1) == 0
    # if zero_mask.any():
        # print("warning: %d zero distance in field_grad" % zero_mask.sum())
    R_unit = R.clone()
    R_unit[~zero_mask] = R[~zero_mask] / R[~zero_mask].norm(dim=-1)[:, None]
    R_unit[zero_mask] = 0
    E = 3 * (p[:, None, :] * R_unit).sum(dim=-1)[:, :, None] * R_unit - p[:, None, :]
    E[zero_mask] = 0
    
    E = E / (R.norm(dim=-1) ** 3 + eps)[:, :, None]
    E_total = E.sum(dim=0) * -1  # field=(-1)*grad -> flip the sign to get the gradient instead of -gradient
    if E_total.isinf().any():
        print("warning: %d inf in field_grad" % E_total.isinf().sum())
    if E_total.isnan().any():
        print("warning: %d nan in field_grad" % E_total.isnan().sum())
    E_total[E_total.isinf()] = 0
    E_total[E_total.isnan()] = 0
    return E_total

# def hoppe_field(sources, means, eps=1e-1):
#     p = sources[:, 3:]
#     R = sources[:, None, :3] - means[None, :, :3]
#     R_norm = R.norm(dim=-1)
#     R_unit = R / (R_norm + eps)[:, :, None]
#     E = 3 * (p[:, None, :] * R_unit).sum(dim=-1)[:, :, None] * R_unit - p[:, None, :]
#     E = E / (R_norm ** 3 + eps)[:, :, None]
#     E_total = E.sum(dim=0) * -1
#     return E_total


def field_edge_calculator_bool(sources, means, if_save=False):
    w,invw = field_edge_calculator(sources, means, if_save)
    if w > 0:
        return 1,-1
    else :
        return -1,1


def field_edge_calculator_count(sources, means, if_save=False):
    w,invw = field_edge_calculator(sources, means, if_save)
    w = sources.shape[0]*means.shape[0]
    if w > 0:
        return w,-w
    else :
        return -w,w

def field_edge_calculator(sources, means, if_save=False):
    def cal_w(S,T):    
        st_E = field_grad(S,T)
        st_interaction = (st_E * T[:,3:]).sum(dim=-1).sum()
        # ts_E = field_grad(T,S)
        # ts_interaction = (ts_E * S[:,3:]).sum(dim=-1).sum()
        # if st_interaction != ts_interaction:
        #     print("%f != %f \t " % (st_interaction, ts_interaction))
        #     print(ts_interaction - st_interaction)
        # 确实不一样，但是差距很小 应该是数值计算的原因
        return (st_interaction * 2) / S.shape[0] * T.shape[0]

    w = cal_w(sources,means) 
    w = w.detach().cpu().numpy()
    invw = w * -1
    return w, invw

# 将nxyz分成两部分，分别计算自相互作用
def self_interaction(nxyz, eps=1e-5):
    assert nxyz.shape[1] == 6
    num = nxyz.shape[0]
    mask = torch.ones(num,dtype=torch.bool)
    mask[torch.randperm(nxyz.shape[0])[:int(num / 2)]]=False
    nxyz1 = nxyz[mask]
    nxyz2 = nxyz[~mask]
    w,_ = field_edge_calculator(nxyz1, nxyz2)
    return w


def self_interaction_all(nxyz, eps=1e-5):
    assert nxyz.shape[1] == 6
    w,_ = field_edge_calculator(nxyz, nxyz)
    return w

def random_self_interaction(nxyz, eps=1e-5):
    assert nxyz.shape[1] == 6
    rand_flip = np.zeros(nxyz.shape[0], dtype=bool)
    rand_flip[np.random.permutation(nxyz.shape[0])[:int(nxyz.shape[0] / 2)]] = True
    rand_n = nxyz.clone()
    rand_n[rand_flip, 3:] *= -1
    w,_ = field_edge_calculator(rand_n, rand_n)
    return w

def reference_field(pc1, pc2):
    with torch.no_grad():
        E = field_grad(pc1, pc2, recursive=True)
        if pc2.shape[1] == 3:
            length = E.norm(dim=-1)
            E[length != 0, :] = E[length != 0, :] / length[length != 0, None]
            pc2 = torch.cat([pc2, E], dim=1)
        else:
            interactions = E * pc2[:, 3:]
            interactions = interactions.sum(dim=-1)
            sign = (interactions >= 0).float() * 2 - 1
            pc2[:, 3:] = pc2[:, 3:] * sign[:, None]

        return pc2


'''
每个patch使用部分点云来代表
'''
def strongest_field_propagation_reps(input_pc, reps, diffuse=False, weights=None):
    input_pc = input_pc.detach()
    with torch.no_grad():
        pts = input_pc

        if weights is not None:
            # factor in the weights for each point by scaling the normals
            weights = weights.clamp(0.1, 1)
            pts[:, 3:] = pts[:, 3:] * weights[:, None]
        device = input_pc.device

        remaining = []

        E = torch.zeros_like(pts[:, :3])
        patches = []
        oriented_pts_mask = torch.zeros(len(pts)).bool()
        non_oriented_pts_mask = torch.zeros(len(pts)).bool()
        for rep, rest in reps:
            remaining.append((rep, rest))
            patches.append(rep)
            non_oriented_pts_mask[rep] = True

        # find the flattest patch to start with
        curv = [util.pca_eigen_values(pts[patch]) for patch in patches]
        min_index = np.array([curv[i][0].cpu() for i in range(len(patches))])
        min_index = np.abs(min_index)
        min_index = np.argmin(min_index)

        # calculate the field from the initial patch
        start_patch, rest_patch = remaining.pop(min_index)
        oriented_pts_mask[start_patch] = True
        non_oriented_pts_mask[start_patch] = False
        E[non_oriented_pts_mask] = field_grad(pts[oriented_pts_mask], pts[non_oriented_pts_mask])

        # prop orientation as long as there are remaining unoriented patches
        while len(remaining) > 0:
            # calculate the interaction between the field and all remaining patches
            interaction = [(E[patch] * pts[patch, 3:]).sum(dim=-1).sum() for patch, rest in remaining]

            # orient the patch with the strongest interaction
            max_interaction_index = torch.tensor(interaction).abs().argmax().item()
            patch, rest_patch = remaining.pop(max_interaction_index)
            # print(f'{patch_index}')
            if interaction[max_interaction_index] < 0:
                pts[patch, 3:] *= -1
                pts[rest_patch, 3:] *= -1
            oriented_pts_mask[patch] = True
            non_oriented_pts_mask[patch] = False

            if diffuse:
                # add the effect of the current patch to *all* other patches
                patch_mask = torch.logical_or(oriented_pts_mask, non_oriented_pts_mask)
                patch_mask[patch] = False
                dE = field_grad(pts[patch], pts[patch_mask])
                E[patch_mask] = E[patch_mask] + dE
            else:
                # add the effect of the current patch only to the *remaining* patches
                dE = field_grad(pts[patch], pts[non_oriented_pts_mask])
                E[non_oriented_pts_mask] = E[non_oriented_pts_mask] + dE

        if diffuse:
            for rep, rest in reps:
                interactions = (E[rep] * pts[rep, 3:]).sum(dim=-1)
                sign = (interactions > 0).float() * 2 - 1
                pts[rep, 3:] = pts[rep, 3:] * sign[:, None]

        E = field_grad(pts[oriented_pts_mask], pts[~oriented_pts_mask])
        interactions = (E * pts[~oriented_pts_mask, 3:]).sum(dim=-1)
        sign = (interactions > 0).float() * 2 - 1
        pts[~oriented_pts_mask, 3:] = pts[~oriented_pts_mask, 3:] * sign[:, None]

        pts = pts.to(device)

        if weights is not None:
            # scale the normal back to unit because of previous weighted scaling
            pts[:, 3:] = pts[:, 3:] / weights[:, None]


# 对每个patch使用
def strongest_field_propagation(pts, patches, all_patches, diffuse=False, weights=None):
    with torch.no_grad():
        if weights is not None:
            # factor in the weights for each point by scaling the normals
            weights = weights.clamp(0.1, 1)
            pts[:, 3:] = pts[:, 3:] * weights[:, None]
        device = pts.device

        # initialize remaining
        remaining = []
        pts_mask = torch.zeros(len(pts)).bool()
        zeros = torch.zeros(len(pts)).bool()
        E = torch.zeros_like(pts[:, :3])
        for i in range(len(all_patches)):
            remaining.append((i, all_patches[i]))

        # find the flattest patch to start with
        curv = [util.pca_eigen_values(pts[patch]) for patch in all_patches]
        min_index = np.array([curv[i][0] for i in range(len(all_patches))])
        min_index = np.abs(min_index)
        min_index = np.argmin(min_index)

        # calculate the field from the initial patch
        _, start_patch = remaining.pop(min_index)
        pts_mask[start_patch] = True
        E[~pts_mask] = field_grad(pts[pts_mask], pts[~pts_mask])

        # prop orientation as long as there are remaining unoriented patches
        while len(remaining) > 0:
            # calculate the interaction between the field and all remaining patches
            interaction = [(E[patch] * pts[patch, 3:]).sum(dim=-1).sum() for i, patch in remaining]

            # orient the patch with the strongest interaction
            max_interaction_index = torch.tensor(interaction).abs().argmax().item()
            patch_index, patch = remaining.pop(max_interaction_index)
            # print(f'{patch_index}')
            if interaction[max_interaction_index] < 0:
                pts[patch, 3:] *= -1
            pts_mask[patch] = True

            if diffuse:
                # add the effect of the current patch to *all* other patches
                patch_mask = zeros.clone()
                patch_mask[patch] = True
                dE = field_grad(pts[patch], pts[~patch_mask])
                E[~patch_mask] = E[~patch_mask] + dE
            else:
                # add the effect of the current patch only to the *remaining* patches
                dE = field_grad(pts[patch], pts[~pts_mask])
                E[~pts_mask] = E[~pts_mask] + dE

        if diffuse:
            for patch in patches:
                patch = patch[1]
                interactions = (E[patch] * pts[patch, 3:]).sum(dim=-1)
                sign = (interactions > 0).float() * 2 - 1
                pts[patch, 3:] = pts[patch, 3:] * sign[:, None]

        pts = pts.to(device)

        if weights is not None:
            # scale the normal back to unit because of previous weighted scaling
            pts[:, 3:] = pts[:, 3:] / weights[:, None]



# 单独使用
def strongest_field_propagation_points(pts: torch.Tensor, diffuse=False, starting_point=0,verbose = False):
        device = pts.device
        pts = pts.cuda()
        indx = torch.arange(pts.shape[0]).to(pts.device)

        E = torch.zeros_like(pts[:, :3])
        visited = torch.zeros_like(pts[:, 0]).bool()
        visited[starting_point] = True
        E[~(indx == starting_point)] += field_grad(pts[starting_point:(starting_point + 1)],
                                                  pts[~(indx == starting_point), :3], eps=1e-6)

        # prop orientation as long as there are remaining unoriented points
        while not visited.all():
            if verbose and torch.sum(visited) == 1:
                draw_field(pts[visited], pts[~visited],field_grad,times=sum(visited))
            # calculate the interaction between the field and all remaining patches
            interaction = (E[~visited] * pts[~visited, 3:]).sum(dim=-1)

            # orient the patch with the strongest interaction
            max_interaction_index = interaction.abs().argmax()
            pts_index = indx[~visited][max_interaction_index]
            # print(f'{patch_index}')
            if interaction[max_interaction_index] < 0:
                pts[pts_index, 3:] *= -1
            visited[pts_index] = True

            E[~(indx == pts_index)] += field_grad(pts[pts_index:(pts_index + 1)],
                                                      pts[~(indx == pts_index), :3], eps=1e-6)

        if diffuse:
            interactions = (E * pts[:, 3:]).sum(dim=-1)
            sign = (interactions > 0).float() * 2 - 1
            pts[:, 3:] = pts[:, 3:] * sign[:, None]

        pts = pts.to(device)
        return pts


# # 返回xie_field.shape = [targt.shape[0],source.shape[0],3],xie_field[i,j]表示第i个target点受到的第j个source产生的能量
# def xie_field(source:torch.Tensor, target: torch.Tensor, eps, max_pts=5000):
#     with torch.no_grad():
#         if source.shape[0] * target.shape[0] > max_pts ** 2:
#             def break_by_source():
#                 mid = int(source.shape[0] / 2)
#                 return torch.cat([xie_field(source[:mid], target, eps, max_pts),
#                                     xie_field(source[mid:], target, eps, max_pts)], dim=1)
                
#             def break_by_target():
#                 mid = int(target.shape[0] / 2)
#                 return torch.cat([xie_field(source, target[:mid], eps, max_pts),
#                                     xie_field(source, target[mid:], eps, max_pts)], dim=0)
#             if source.shape[0] > target.shape[0]:
#                 return break_by_source()
#             else:
#                 return break_by_target()
#         R = source[None,:,:3] - target[:,None,:3] # M x N x 3, 表示第M个target点到第N个source点的距离向量
#         R_norm = R.norm(dim=-1) # 
#         zero_mask = R_norm == 0
#         normal_s = source[:, 3:] 
#         horizental_distant = torch.cross(normal_s[None,:,:], R).norm(dim=-1)  / normal_s.norm(dim=-1)[None,:]
#         Gussian = torch.zeros_like(R_norm)
#         # h_zero_mask = horizental_distant == 0
#         # Gussian = torch.exp(-horizental_distant ** 2 / (2 * eps ** 2))
#         # Gussian = torch.exp(-R_norm ** 2 / (2 * eps ** 2)) * 100
#         # Gussian[~h_zero_mask] = Gussian[~h_zero_mask] / (horizental_distant[~h_zero_mask] ** 3)
#         # Gussian[~zero_mask] = Gussian[~zero_mask] / (R_norm[~zero_mask] ** 3)
#         Gussian[~zero_mask] = torch.ones_like(Gussian[~zero_mask])/ ((R_norm[~zero_mask] + horizental_distant[~zero_mask]) ** 3)
        
#         R_unit = R.clone()
#         R_unit[~zero_mask] = R[~zero_mask] / R[~zero_mask].norm(dim=-1)[:, None]
#         normal_s = source[:, 3:] 
#         semi_normal_s = 2 * (normal_s * R_unit).sum(dim=-1)[:, :, None] * R_unit - normal_s
#         ref_normal_s = semi_normal_s * -1
#     return ref_normal_s * Gussian[:, :, None]


# 返回xie_field.shape = [targt.shape[0],source.shape[0],3],xie_field[i,j]表示第i个target点受到的第j个source产生的能量
def xie_field(source:torch.Tensor, target: torch.Tensor, eps, max_pts=5000):
    with torch.no_grad():
        if source.shape[0] * target.shape[0] > max_pts ** 2:
            def break_by_source():
                mid = int(source.shape[0] / 2)
                return torch.cat([xie_field(source[:mid], target, eps, max_pts),
                                    xie_field(source[mid:], target, eps, max_pts)], dim=1)
                
            def break_by_target():
                mid = int(target.shape[0] / 2)
                return torch.cat([xie_field(source, target[:mid], eps, max_pts),
                                    xie_field(source, target[mid:], eps, max_pts)], dim=0)
            if source.shape[0] > target.shape[0]:
                return break_by_source()
            else:
                return break_by_target()
        R = source[None,:,:3] - target[:,None,:3] # M x N x 3, 表示第M个target点到第N个source点的距离向量
        R_norm = R.norm(dim=-1) # 
        zero_mask = R_norm == 0
        normal_s = source[:, 3:] 
        R_unit = R.clone()
        R_unit[~zero_mask] = R[~zero_mask] / R[~zero_mask].norm(dim=-1)[:, None]
        normal_s = source[:, 3:] 
        ref_normal_s  = normal_s - 3 * (normal_s * R_unit).sum(dim=-1)[:, :, None] * R_unit 
        ref_normal_s[~zero_mask] /= ((R_norm[~zero_mask]) ** 3)[:,None]
    return ref_normal_s


def draw_field(source:torch.Tensor, target: torch.Tensor, field_cacular,opt = 'save', times = 0,*args, **kwargs):
    field = field_cacular(source, target, *args, **kwargs)

    # todo 不是field_grad的shape是反过来的
    if not field_cacular.__name__ == 'field_grad':
        field = field.sum(dim=-2)
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    xyz = np.concatenate([target.cpu().numpy(), source.cpu().numpy()], axis=0)
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz[:len(target), 3:] = field.cpu().numpy()
    pc.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pc.normals = o3d.utility.Vector3dVector(xyz[:, 3:])
    colors = np.zeros_like(xyz[:, :3])
    colors[:len(target), 1] = 1
    colors[len(target):, 0] = 1
    colors = np.asarray(colors, dtype=np.float64)
    pc.colors = o3d.utility.Vector3dVector(colors)
    
    if opt == 'save':
        import os
        func_name = field_cacular.__name__
        floder = "temp/field/"
        os.makedirs(floder, exist_ok=True)
        path = '%s_%d.ply' % (func_name, times)
        o3d.io.write_point_cloud(floder + path, pc)
    elif opt == 'show':
        o3d.visualization.draw_geometries([pc])
    else:
        print("opt error")
    


# intersaction[j] = (n - 2 r * cos(theta) )* n * gaussian = n - 2 r * (n_norm - n * r) * gaussian
# source: N x 6, target: M x 6 
# eps: 高斯核参数 越大越平滑
# return: M x N
def xie_intersaction(source: torch.Tensor, target: torch.Tensor, eps):
    with torch.no_grad():
        ref_normal_s = xie_field(source, target, eps=eps)
        intersaction  = ( ref_normal_s * target[:,None, 3:] ).sum(dim=-1)
        if intersaction.isnan().any():
            print("warning: %d nan in xie_intersaction" % intersaction.isnan().sum())
        if intersaction.isinf().any():
            print("warning: %d inf in xie_intersaction" % intersaction.isinf().sum())
        intersaction[intersaction.isnan()] = 0
        intersaction[intersaction.isinf()] = 0
    return intersaction


def xie_distance(source: torch.Tensor, target: torch.Tensor, eps):
    R = source[None,:,:3] - target[:,None,:3] 
    xie_distance = source[None,:,3:] * R
    xie_distance = xie_distance.norm(dim=-1).sum(dim=-1)
    return xie_distance
    




def xie_propagation_points_plus(pts: torch.Tensor, eps, diffuse=False, starting_point=0):
    pts = strongest_field_propagation_points(pts)
    return xie_propagation_points(pts, eps, diffuse, starting_point=starting_point)

def xie_propagation_points(pts: torch.Tensor, eps, diffuse=False, starting_point=0,verbose = False):
    def diffuse_field(pts, eps,verbose = False,times = 0):
        interactions = xie_intersaction(pts, pts, eps=eps).sum(dim=-1)
        sign = (interactions > 0).float() * 2 - 1
        pts[:, 3:] = pts[:, 3:] * sign[:, None]
        if verbose:
            print("%d flipped in diffuse" % (sign == -1).sum()) 
            draw_field(pts, pts, xie_field, eps=eps, times=times)  
    
    device = pts.device
    indx = torch.arange(pts.shape[0]).to(pts.device)
    visited = torch.zeros_like(pts[:, 0]).bool()
    interactions = torch.zeros(len(pts)).to(pts.device) # 当前visited点对所有点的影响。shape: N*1
    visited[starting_point] = True
    while not visited.all():
        interactions[~visited] += xie_intersaction(pts[visited], pts[~visited], eps=eps).sum(dim=-1)
        if torch.sum(visited) % 10 == 1 and verbose:
            draw_field(pts[visited], pts[~visited], xie_field, eps=eps,times=sum(visited))
            # draw_field(pts[visited], pts[~visited], field_grad, eps=eps,times=sum(visited))
            
        pts_index = indx[~visited][interactions[~visited].argmax()]
        if interactions[pts_index] < 0:
            pts[pts_index, 3:] *= -1
        visited[pts_index] = True
    
    if diffuse:
        diffuse_field(pts, eps)
    pts = pts.to(device)

'''
order : T x N, 表示第i次传播的顺序
return bool tensor, N x T, 表示第i个点是否在第j次是否被翻转
'''
def xie_propagation_points_in_order(pts: torch.Tensor, eps, order, diffuse=False,verbose=False,points_weight = None):
    with torch.no_grad():
        MyTimer = util.timer_factory()
        order = torch.tensor(order).to(pts.device)
        order.data = order.data.long()
        T,N = order.shape
        
        pointWeight = torch.ones(len(pts)).to(pts.device)
        
        # 每个点的权重被设置为其第k远邻的距离
        
        with MyTimer("prepare"):
            interactions = torch.zeros(T,N).to(pts.device).type(pts.dtype) # 当前visited点对所有点的影响。shape: T x N
            interaction_mat = xie_intersaction(pts, pts, eps=eps) # N x N, 表示第i个点受到的来自第j个点的电场
            if points_weight is not None:
                interaction_mat = interaction_mat * pointWeight[None,:]
            visited = torch.zeros_like(order).bool() # T x N
            weights = visited.clone().type(pts.dtype) # T x N 表示第i次传播时，第j个点翻转的状态
            
        rg = torch.arange(T).to(pts.device)
        with MyTimer("propagation"):
            for i in range(N):
                idx = order[:,i]
                visited[rg,idx] = True
                interactions[rg,idx] = torch.sum(interaction_mat[idx] * weights, dim=-1)
                # print(weights.dtype)
                weights[rg,idx] = torch.where(interactions[rg,idx] < 0, -1.0, 1.0).type(pts.dtype)
                        
        if diffuse:
            with MyTimer("diffuse"):
                if interaction_mat.shape[0] < 5000:
                    interactions  = (interaction_mat[None,:,:] * weights[:,None,:]).sum(dim=-1)
                else:
                    for i in range(T):
                        interactions[i] = (interaction_mat * weights[i][None,:]).sum(dim=-1)
    torch.cuda.empty_cache()
    return (interactions<0)


def cal_loss(x,A,B):
    n = len(x)
    # x = np.array(x, dtype=int)
    assert A.shape == (n,n)
    assert B.shape == (n,n)
    obj = 0
    for i in range(n):
        for j in range(n):
            obj += A[i,j]*(1 - (x[i]-x[j])*(x[i]-x[j])) + B[i,j]*(x[i]- x[j])*(x[i] - x[j])
    return obj

import gurobipy as gp
def MIQP(A,B):
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]
    # Create a new model
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    # Create variables
    n = len(A)
    x = m.addVars(n, vtype=gp.GRB.BINARY, name="x")
    # Set objective
    obj = gp.QuadExpr()
    obj -= cal_loss(x,A,B)
    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # find the optimal solution
    m.optimize()
    res = np.zeros(n)
    
    # print('Obj: %g' % m.objVal)
    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))
    # print('Optimal solution found')
    
    print('Obj: %g' % m.objVal)
    for i in range(n):
        res[i] = x[i].x
    return res


import graph    
'''
以BFS的顺序传播
times: 传播次数;最后投票;默认为1,即只从starting_point开始传播一次;如果times>1,则随机从pts中再选择times个点作为starting_point。必须是正奇数
对齐策略:要求任意两个点的翻转状态的异或结果最小
k: 生成图的k近邻
treshold: 生成图的treshold
'''
def xie_propagation_points_onbfstree(pts: torch.Tensor, eps, diffuse=False, starting_point=0,verbose = False,k=10,treshold=0.1,times = 1,use_weight = False):
    assert times % 2 == 1 and times > 0
    MyTimer = util.timer_factory()
    with MyTimer("Generate Graph"):
        starting_points = [starting_point]
        while len(np.unique(starting_points)) < times:
            t = np.random.randint(0,pts.shape[0])
            if t not in starting_points:
                starting_points.append(t)
        
        xyz = pts[:,:3].cpu().numpy()
        G,mean_k_dist = graph.getLinkedListGraphfromPc(xyz, k, treshold)
        if use_weight:
            pointWeight = mean_k_dist
        else:
            pointWeight = None
        
        def cal_w(order1,order2):
            w = abs(torch.sum(order1^order2))
            invw = len(order1) - w
            return w,invw    
    
        cnts = torch.zeros(len(pts),dtype=torch.int).to(pts.device) 
        # all_flipstatus = torch.zeros([len(pts),times],dtype=torch.bool).to(pts.device)
    
    orders = np.zeros([times,len(pts)],dtype=int)
    with MyTimer("Multi BFS"):
        for i in range(times):
            st = starting_points[i]
            orders[i] = G.get_bfs_route(st)
           
    # for i in range(times):
    #     with MyTimer("xie_propagation_points_in_order times = %d" % times):
    #         all_flipstatus[:,i] = xie_propagation_points_in_order(pts.clone(), eps, [orders[i]], diffuse,verbose=False)[0]

    with MyTimer("xie_propagation_points_in_order times = %d" % times):
        all_flipstatus = xie_propagation_points_in_order(pts.clone(), eps, orders, diffuse,verbose=False,points_weight=pointWeight).T

    with MyTimer("Vote"):
        A = torch.zeros([times,times],dtype=torch.float).to(pts.device)
        B = torch.zeros([times,times],dtype=torch.float).to(pts.device)
        for i in range(times):
            for j in range(times):
                A[i,j],B[i,j] = cal_w(all_flipstatus[:,i],all_flipstatus[:,j])
        status = MIQP(A.cpu().numpy(),B.cpu().numpy())
        status = torch.tensor(status,dtype=torch.bool).to(pts.device)
        for i in range(times):
            all_flipstatus[:,i] = all_flipstatus[:,i] ^ status[i]
            cnts += all_flipstatus[:,i].int()
        for i in range(len(pts)):
            if cnts[i] > times/2:
                pts[i,3:] *= -1
    return cnts > times/2