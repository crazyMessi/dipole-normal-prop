import field_utils
import util
import numpy as np
from pathlib import Path
import torch
from graph import *
import open3d as o3d

base_path = "D:/Documents/zhudoongli/CG/project/NormalEstimation/dipole-normal-prop"
# pc_name = "scene0000_102201_gt25.ply"
pc_name = "scene004_102202_gt75.ply"

input_pc_path = base_path + "/data/" + pc_name
output_path = base_path + "/data/output/"
if not Path(output_path).exists():
    Path(output_path).mkdir(parents=True)
    
def _st_propagation_points(input_pc):
    input_pc, transform = util.Transform.trans(input_pc)
    field_utils.strongest_field_propagation_points(input_pc, True, starting_point=0)
    if field_utils.measure_mean_potential(input_pc) < 0:
        input_pc[:, 3:] *= -1
    input_pc = transform.inverse(input_pc)
    return input_pc

def single_dipole(pc_path):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        pc = o3d.io.read_point_cloud(pc_path)
        xyz = np.asarray(pc.points)
        input_pc = torch.tensor(xyz, dtype=torch.float32, device=device)
        normals = np.asarray(pc.normals)
        gt_pc = torch.tensor(torch.cat([torch.tensor(xyz, dtype=torch.float32, device=device), torch.tensor(normals, dtype=torch.float32, device=device)], dim=1), dtype=torch.float32, device=device)
    input_pc = util.estimate_normals(input_pc, max_nn=30)    
    input_pc =  _st_propagation_points(input_pc)
    util.draw_pc(input_pc, path=Path(output_path + "/simple_result.ply"))
    # 如果有gt_pc,计算误差
    if gt_pc.shape[1] == 6:
        loss = util.cal_loss(gt_pc,input_pc)
        print("loss:",loss)
        return loss

def graph_dipole_api(xyz_data,config):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    input_pc = util.npxyz2tensor(xyz_data).to(device)
    input_pc = util.estimate_normals(input_pc, max_nn=config['max_nn'])
    input_pc, transform = util.Transform.trans(input_pc)
    
    if config['divide_method'] == 'grid_partition':

        G,index = util.divide_pc_to_graph(input_pc, 
                                        n_part=config["n_part"],
                                        min_patch=config["min_patch"],
                                        edge_calculator=field_utils.field_edge_calculator,
                                        point_estimator=_st_propagation_points)
    
    elif config['divide_method'] == 'ncut_partition':
        G,index = util.divide_pc_by_ncut(input_pc,
                                         k_neighbors=config["k_neighbors"],
                                         mininum_rate=max(config["mininum_rate"],config["min_patch"] / len(input_pc) ),
                                         edge_calculator=field_utils.field_edge_calculator,
                                         point_estimator=_st_propagation_points)
    else:   
        print("Error: no such divide method")
        return 
    
    input_pc = transform.inverse(input_pc)
    A,B = G.to_matrix()
    flip = MIQP(A,B)
    for i in range(len(flip)):
        if flip[i] == 1:
            input_pc[index[i],3:] *= -1
    return input_pc.cpu().numpy()      

def graph_dipole(pc_path, use_ncut=True):
    # load data
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        pc = o3d.io.read_point_cloud(pc_path)
        xyz = np.asarray(pc.points)
        normals = np.asarray(pc.normals)
        input_pc = torch.tensor(xyz, dtype=torch.float32, device=device)
        if normals.shape[0] == xyz.shape[0]:
            xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
            normals = torch.tensor(normals, dtype=torch.float32, device=device)
            gt_pc = torch.tensor(torch.cat([xyz.clone(), normals.clone()], dim=1), dtype=torch.float32, device=device)
    print(input_pc.shape)
    
    input_pc, transform = util.Transform.trans(input_pc)
    
    with MyTimer('estimating normals'):
        input_pc = util.estimate_normals(input_pc, max_nn=30)
        # input_pc = util.meshlab_estimate_normal(input_pc)
    
    with MyTimer('divide to graph and rectify patches'):    
        if not use_ncut:
            G,index = util.divide_pc_to_graph(input_pc, 
                                            n_part=10,
                                            min_patch=0,
                                            edge_calculator=field_utils.field_edge_calculator,
                                            point_estimator=_st_propagation_points)
        else:
            G,index = util.divide_pc_by_ncut(input_pc,
                                             k_neighbors=30,
                                             mininum_rate=1.0 / 10,
                                             edge_calculator=field_utils.field_edge_calculator,
                                             point_estimator=_st_propagation_points)
        
                                             
     
    labels = torch.zeros(input_pc.shape[0])
    for i in range(len(index)):
        labels[index[i]] = i
        
    input_pc = transform.inverse(input_pc)
    # 输出初始结果
    util.draw_pc(input_pc, path=Path(output_path + "/initial_result.ply"),labels=labels)
    
    with MyTimer('flip patches'):    
        A,B = G.to_matrix()
        flip = MIQP(A,B)
        for i in range(len(flip)):
            if flip[i] == 1:
                input_pc[index[i],3:] *= -1
    if normals.shape[0] == xyz.shape[0]:    
        g_pc = GraphPC(G,input_pc,index,gt_pc)
        g_pc.print_metrics()
        node_labels = g_pc.get_node_flip_status()
        edge_labels = g_pc.get_edge_correctness()
        # 从tensor转换为numpy
        node_labels = [int(x.cpu().numpy()) for x in node_labels]
        edge_labels = [int(x.cpu().numpy()) for x in edge_labels]
        util.draw_topology(G,input_pc,index,nodelabel=node_labels,edgelabel=edge_labels,path=output_path + "/colored_topology.ply")
        g_pc.save_wrong_edge(output_path + "/wrong_edge")
        g_pc.save_all_edge(output_path + "/all_edge")
    
    util.draw_pc(input_pc, path=Path(output_path + "/final_result.ply"),labels=labels)
    util.draw_topology(G,input_pc,index,path=output_path + "/topology.ply")
    
    # 如果有gt_pc,计算误差
    if normals.shape[0] == xyz.shape[0]:
        loss = util.cal_loss(gt_pc,input_pc)
        print("loss:",loss)
        return loss

def run_floder(floder,exp_name):
    pc_list = os.listdir(floder)
    log = open("temp/%s.log" % exp_name,"w")
    for pc in pc_list:
        if pc[-3:] == "ply" and pc.find("gt") != -1:
            print("processing:",pc)
            g_loss = graph_dipole(floder + "/" + pc)
            s_loss = single_dipole(floder + "/" + pc)
            g_loss = str(g_loss)
            s_loss = str(s_loss)
            log.write(pc + "g_loss: " + g_loss + "s_loss: " + s_loss + "\n")
            print("=============================================")



if __name__ == '__main__':
    MyTimer = util.timer_factory()
    with MyTimer('graph_dipole'):
        graph_dipole(input_pc_path)
        
    print("=============================================")

    with MyTimer('single_dipole'):
        single_dipole(input_pc_path)
    