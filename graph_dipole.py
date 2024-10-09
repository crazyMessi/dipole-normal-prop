import field_utils
import util
import numpy as np
from pathlib import Path
import torch
from graph import *

base_path = "D:/Documents/zhudoongli/CG/project/NormalEstimation/dipole-normal-prop"
pc_name = "flower.xyz"
input_pc_path = base_path + "/data/" + pc_name
output_path = base_path + "/data/output/"
if not Path(output_path).exists():
    Path(output_path).mkdir(parents=True)
    



def graph_dipole(pc_path):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        input_pc = util.xyz2tensor(open(Path(pc_path), 'r').read(), append_normals=False).to(device)
    print(input_pc.shape)

    input_pc, transform = util.Transform.trans(input_pc)

    with MyTimer('estimating normals'):
        input_pc = util.meshlab_estimate_normal(input_pc)
    with MyTimer('divide patches'):    
        G,index = util.divide_pc_to_graph(input_pc, n_part=100, min_patch=21,edge_calculator=field_utils.field_edge_calculator,point_estimator=field_utils.strongest_field_propagation_points)
        # G,index = util.divide_pc_to_graph(input_pc, n_part=100, min_patch=21,edge_calculator=field_utils.field_edge_calculator)
        print("number of patches: ",len(G.V))
        
    labels = torch.zeros(input_pc.shape[0])
    for i in range(len(index)):
        labels[index[i]] = i
        
    
    # 输出初始结果
    util.draw_pc(input_pc, path=Path(output_path + "/initial_result.ply"),labels=labels)
    
    
    with MyTimer('flip patches'):    
        A,B = G.to_matrix()
        flip = MIQP(A,B)
        for i in range(len(flip)):
            if flip[i] == 1:
                input_pc[index[i],3:] *= -1
    
    util.draw_pc(input_pc, path=Path(output_path + "/final_result.ply"),labels=labels)
    util.draw_topology(G,input_pc,index,path=output_path + "/topology.ply")


if __name__ == '__main__':
    graph_dipole(input_pc_path)
    