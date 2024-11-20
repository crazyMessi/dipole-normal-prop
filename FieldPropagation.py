# import torch
# import numpy as np
# import util
# import pdb


# class FieldCalculator:
#     def __init__(self):
#         pass
    
    
#     # 给定N*3的target和M*3的source，计算N*M的field_mat, field_mat[i,j]表示第i个target点处收到第j个source点的field
#     def field_mat(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         pass
    
#     # 给定N*3的target和M*3的source，计算N*3的field, field[i]表示第i个target点处的field
#     def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         pass
    
    
#     # 给定N*3的source和M*3的target，计算N*M的能量(标量)，energy[i,j]表示第i个source点对第j个target点产生的能量
#     def intersaction(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return (self.__call__(source, target) * target[:, 3:]).sum(dim=-1)    
    
    
# class XieFieldCalculator(FieldCalculator):
#     def __init__(self):
#         pass
    
#     def field_mat(source:torch.Tensor, target: torch.Tensor):
#         R = source[None,:,:3] - target[:,None,:3] # M x N x 3, 表示第M个target点到第N个source点的距离向量
#         R_norm = R.norm(dim=-1) # 
#         zero_mask = R_norm == 0
#         normal_s = source[:, 3:] 
#         horizental_distant = torch.cross(normal_s[None,:,:], R).norm(dim=-1)  / normal_s.norm(dim=-1)[None,:]
#         distance_decay = torch.zeros_like(R_norm)
#         distance_decay[~zero_mask] = torch.ones_like(distance_decay[~zero_mask])/ ((R_norm[~zero_mask] + horizental_distant[~zero_mask]) ** 3)
        
#         R_unit = R.clone()
#         R_unit[~zero_mask] = R[~zero_mask] / R[~zero_mask].norm(dim=-1)[:, None]
#         semi_normal_s = 2 * (normal_s * R_unit).sum(dim=-1)[:, :, None] * R_unit - normal_s
#         ref_normal_s = semi_normal_s * -1
        
#         if ref_normal_s.isnan().any():
#             print("warning: %d nan in field_mat" % ref_normal_s.isnan().sum())
#             ref_normal_s[ref_normal_s.isnan()] = 0
#         if ref_normal_s.isinf().any():
#             print("warning: %d inf in field_mat" % ref_normal_s.isinf().sum())
#             ref_normal_s[ref_normal_s.isinf()] = 0  
#         return ref_normal_s * distance_decay[:, :, None]
    
#     def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.field_mat(source, target).sum(dim=1)

# class DipoleFieldCalculator(FieldCalculator):
#     def __init__(self,eps):
#         self.eps = eps
    
#     def field_mat(source: torch.Tensor, target: torch.Tensor):
#         R = target[:, None, :3] - source[None, :, :3]
#         p = source[:, 3:]
#         zero_mask = R.norm(dim=-1) == 0
#         R_unit = R.clone()
#         R_unit[~zero_mask] = R[~zero_mask] / R[~zero_mask].norm(dim=-1)[:, None]
#         R_unit[zero_mask] = 0
#         E = 3 * (p[None, :, :] * R_unit).sum(dim=-1) * R_unit - p[None, :, :]
#         E[zero_mask] = 0
#         E = E / (R.norm(dim=-1) ** 3)[:, :, None]
#         if E.isinf().any():
#             print("warning: %d inf in dipole_field" % E.isinf().sum())
#         if E.isnan().any():
#             print("warning: %d nan in dipole_field" % E.isnan().sum())
#         E[E.isinf()] = 0
#         E[E.isnan()] = 0
#         return E
    
#     def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.field_mat(source, target).sum(dim=1)


# # class Propagator:
# #     def __init__(self, field_calculator: FieldCalculator):
# #         self.field_calculator = field_calculator
    
# #     def strongest_field_propagation_points(self,pts: torch.Tensor, diffuse=False, starting_point=0,verbose = False):
# #             device = pts.device
# #             pts = pts.cuda()
# #             indx = torch.arange(pts.shape[0]).to(pts.device)

# #             E = torch.zeros_like(pts[:, :3])
# #             visited = torch.zeros_like(pts[:, 0]).bool()
# #             visited[starting_point] = True
# #             E[~(indx == starting_point)] += self.field_calculator(pts[~(indx == starting_point),pts[starting_point:(starting_point + 1)], :3])

# #             # prop orientation as long as there are remaining unoriented points
# #             while not visited.all():
# #                 if verbose and torch.sum(visited) == 1:
# #                     draw_field(pts[visited], pts[~visited],field_grad,times=sum(visited))
# #                 # calculate the interaction between the field and all remaining patches
# #                 interaction = (E[~visited] * pts[~visited, 3:]).sum(dim=-1)

# #                 # orient the patch with the strongest interaction
# #                 max_interaction_index = interaction.abs().argmax()
# #                 pts_index = indx[~visited][max_interaction_index]
# #                 # print(f'{patch_index}')
# #                 if interaction[max_interaction_index] < 0:
# #                     pts[pts_index, 3:] *= -1
# #                 visited[pts_index] = True

# #                 E[~(indx == pts_index)] += field_grad(pts[pts_index:(pts_index + 1)],
# #                                                         pts[~(indx == pts_index), :3], eps=1e-6)

# #             if diffuse:
# #                 interactions = (E * pts[:, 3:]).sum(dim=-1)
# #                 sign = (interactions > 0).float() * 2 - 1
# #                 pts[:, 3:] = pts[:, 3:] * sign[:, None]

# #             pts = pts.to(device)
# #             return pts
