import field_utils
import util
import numpy as np
from pathlib import Path
import torch
from graph import *
import open3d as o3d

base_path = "D:/Documents/zhudoongli/CG/project/NormalEstimation/dipole-normal-prop/"
pc_name = "scene0000_gt122.ply"

input_pc_path = base_path + "/data/hard/" + pc_name
# input_pc_path = "D:\WorkData\ipsr_explore\input\someseg/SceneNN_076_res334.ply"

output_path = base_path + "/data/output/"

CUDA_VISIBLE_DEVICES = 7

if not Path(output_path).exists():
    Path(output_path).mkdir(parents=True)

def propagate_points_file(input_pc,propagate_func, *args, **kwargs):
    input_pc, transform = util.Transform.trans(input_pc)
    propagate_func(input_pc, *args, **kwargs)
    if field_utils.measure_mean_potential(input_pc) < 0:
        input_pc[:, 3:] *= -1
    input_pc = transform.inverse(input_pc)
    return input_pc

def st_propagation_points_file(input_pc,verbose=True):
    return propagate_points_file(input_pc,field_utils.strongest_field_propagation_points, diffuse=True, starting_point=0,verbose=verbose)

def xie_propagation_points_file(input_pc,eps = 1e-2,verbose=True):
    return propagate_points_file(input_pc,field_utils.xie_propagation_points, eps=eps, diffuse=True, starting_point=0,verbose=verbose)

def xie_tree_propagation_points_file(input_pc,eps = 1e-2,verbose=True,times=1):
    return propagate_points_file(input_pc,field_utils.xie_propagation_points_onbfstree, eps=eps, diffuse=True, starting_point=0,verbose=verbose,times=times)



def single_propagate_file(pc_path,verbose=True, use_origin_normal=False, propagation_method=st_propagation_points_file,*args, **kwargs):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    # device = torch.device('cpu')
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        pc = o3d.io.read_point_cloud(pc_path)
        xyz = np.asarray(pc.points)
        input_pc = torch.tensor(xyz, dtype=torch.float32, device=device)
        normals = np.asarray(pc.normals)
        gt_pc = torch.cat([torch.tensor(xyz, dtype=torch.float32, device=device), torch.tensor(normals, dtype=torch.float32, device=device)], dim=1)
    if not use_origin_normal:
        input_pc = util.estimate_normals(input_pc, max_nn=30)    
    else:
        input_pc = gt_pc.clone()
    
    input_pc =  propagation_method(input_pc,verbose=verbose,*args, **kwargs)
    if verbose:
        util.draw_pc(input_pc, path=Path(output_path + "/single_%s_result.ply"% propagation_method.__name__))
    # 如果有gt_pc,计算误差
    if gt_pc.shape[1] == 6:
        loss = util.cal_loss(gt_pc,input_pc)
        print("loss:",loss)
        return loss

def graph_dipole_server_api(xyz_data,config):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    input_pc = util.npxyz2tensor(xyz_data).to(device)
    input_pc = util.estimate_normals(input_pc, max_nn=config['max_nn'])
    input_pc, transform = util.Transform.trans(input_pc)
    
    if config['divide_method'] == 'grid_partition':

        G,index = util.divide_pc_to_graph(input_pc, 
                                        n_part=config["n_part"],
                                        min_patch=config["min_patch"],
                                        edge_calculator=field_utils.field_edge_calculator,
                                        point_estimator=st_propagation_points_file)
    
    elif config['divide_method'] == 'ncut_partition':
        G,index = util.divide_pc_by_ncut(input_pc,
                                         k_neighbors=config["k_neighbors"],
                                         mininum_rate=max(config["mininum_rate"],config["min_patch"] / len(input_pc) ),
                                         edge_calculator=field_utils.field_edge_calculator,
                                         point_estimator=st_propagation_points_file)
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

def graph_dipole(pc_path, use_ncut=True, verbose=True):
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
                                            point_estimator=st_propagation_points_file)
        else:
            G,index = util.divide_pc_by_ncut(input_pc,
                                             k_neighbors=30,
                                             mininum_rate=1.0 / 10,
                                             edge_calculator=field_utils.field_edge_calculator,
                                             point_estimator=st_propagation_points_file)
        
                                             
     
    labels = torch.zeros(input_pc.shape[0])
    for i in range(len(index)):
        labels[index[i]] = i
        
    input_pc = transform.inverse(input_pc)
    # 输出初始结果
    if verbose:
        util.draw_pc(input_pc, path=Path(output_path + "/initial_result.ply"),labels=labels)
     
    with MyTimer('flip patches'):    
        A,B = G.to_matrix()
        flip = MIQP(A,B)
        for i in range(len(flip)):
            if flip[i] == 1:
                input_pc[index[i],3:] *= -1
        
    if normals.shape[0] == xyz.shape[0]:    
        g_pc = GraphPC(G,input_pc,index,gt_pc,flip_status=torch.tensor(flip,dtype=torch.float32,device=device))
        g_pc.print_metrics()
        node_labels = g_pc.get_node_flip_status()
        edge_labels = g_pc.get_edge_correctness()
        # 从tensor转换为numpy
        node_labels = [int(x.cpu().numpy()) for x in node_labels]
        edge_labels = [int(x.cpu().numpy()) for x in edge_labels]
        if verbose:
            util.draw_topology(G,input_pc,index,nodelabel=node_labels,edgelabel=edge_labels,path=output_path + "/colored_topology.ply")
            g_pc.save_wrong_edge(output_path + "/wrong_edge")
            g_pc.save_all_edge(output_path + "/all_edge")

    if verbose:
        util.draw_pc(input_pc, path=Path(output_path + "/final_result.ply"),labels=labels)
        util.draw_topology(G,input_pc,index,path=output_path + "/topology.ply")
    

    # 如果有gt_pc,计算误差
    if normals.shape[0] == xyz.shape[0]:
        loss = util.cal_loss(gt_pc,input_pc)
        print("loss:",loss)
        return loss

import threading


def run_file(file) -> str:
    printmsg = "file:%s\t" % file
    MyTimer = util.timer_factory() 
    
    with MyTimer('xie on tree'):
        # gt_tree_xie_loss = str(single_propagate_file(file,use_origin_normal=True,propagation_method=xie_tree_propagation_points_file))
        # print("\n")
        tree_xie_loss = str(single_propagate_file(file,use_origin_normal=False,propagation_method=xie_tree_propagation_points_file,times=5))
        print("\n")        
        # printmsg += "gt_tree_xie_loss:%s\t" % gt_tree_xie_loss
        printmsg += "tree_xie_loss:%s\t" % tree_xie_loss
    print("\n")
    
    with MyTimer('xie dipole'):
        # gt_xie_loss = str(single_propagate_file(file,use_origin_normal=True,propagation_method=xie_propagation_points_file))
        # print("\n")
        
        xie_loss = str(single_propagate_file(file,use_origin_normal=False,propagation_method=xie_propagation_points_file))
        print("\n")
    
        # printmsg += "gt_xie_loss:%s\t" % gt_xie_loss
        printmsg += "xie_loss:%s\t" % xie_loss
    print("\n")
        
    with MyTimer('st dipole'):
        # gt_dipole_loss = str(single_propagate_file(file,use_origin_normal=True,propagation_method=st_propagation_points_file))
        # print("\n")
        
        dipole_loss = str(single_propagate_file(file,use_origin_normal=False,propagation_method=st_propagation_points_file))
        print("\n")
        # printmsg += "gt_dipole_loss:%s\t" % gt_dipole_loss
        printmsg += "dipole_loss:%s\t" % dipole_loss
    return printmsg


def run_floder(floder,exp_name):
    pc_list = os.listdir(floder)
    log = open("temp/%s.log" % exp_name,"w")
    # lock
    lock = threading.Lock()
    threads = []

    def single_handle(filename):
        if filename[-3:] == "ply" and filename.find("gt") != -1:
            msg = run_file(floder + filename)
            lock.acquire()
            print("=============================================")
            print(msg)
            log.write(msg + "\n")
            lock.release()

    for pc in pc_list:
        if pc[-3:] != "ply" or pc.find("gt") == -1:
            continue
        # 创建线程
        t = threading.Thread(target=single_handle,args=(pc,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    log.close()


if __name__ == '__main__':
    MyTimer = util.timer_factory()
    # run_floder("D:\Documents/zhudoongli\CG\project/NormalEstimation/dipole-normal-prop/data/hard/","hard")  
    run_file(input_pc_path)
       
    # with MyTimer('graph_dipole'):
    #     graph_dipole(input_pc_path)
    # run_file(input_pc_path) 
    