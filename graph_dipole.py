import field_utils
import util
import numpy as np
from pathlib import Path
import torch
from graph import *
import open3d as o3d

base_path = "D:/Documents/zhudoongli/CG/project/NormalEstimation/dipole-normal-prop/"
# pc_name = "scene0000_gt122.ply"
pc_name = "93001_scene0037_00_vh_clean_2_gt67.ply"

input_pc_path = base_path + "/data/hard/" + pc_name
input_pc_path = "D:\Documents/zhudoongli\CG\project/NormalEstimation/dipole-normal-prop/data/gt_test_2/scene0054_102201_it_10_gt153.ply"
# input_pc_path = "D:\WorkData\ipsr_explore\input/big_segments/scene0465_00_vh_clean_2_gt11.ply"

pc_name = input_pc_path.split("/")[-1]
pc_name = ".".join(pc_name.split(".")[:-1])

output_path = base_path + "/data/output/" + pc_name + "/"

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

def xie_tree_propagation_points_file(input_pc,eps = 1e-2,verbose=True,times=1,use_pw=False):
    return propagate_points_file(input_pc,field_utils.xie_propagation_points_onbfstree, eps=eps, diffuse=True, starting_point=0,verbose=verbose,times=times,use_pw=use_pw,knn_mask=-1)



def single_propagate_file(pc_path,verbose=False, use_origin_normal=False, propagation_method=st_propagation_points_file,
                          gt_path=None,
                          *args, **kwargs):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    # device = torch.device('cpu')
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        pc = o3d.io.read_point_cloud(pc_path)
        
        xyz = np.asarray(pc.points)
        input_pc = torch.tensor(xyz, dtype=torch.float32, device=device)
        normals = np.asarray(pc.normals)
        ori_pc = torch.cat([torch.tensor(xyz, dtype=torch.float32, device=device), torch.tensor(normals, dtype=torch.float32, device=device)], dim=1)
    
        if gt_path != None:
            gt_pc = o3d.io.read_point_cloud(gt_path)
            gt_xyz = np.asarray(gt_pc.points)
            diff = np.linalg.norm(xyz - gt_xyz)
            if diff > 1e-3:
                print("Error: gt and input pc not match")
                assert False
                return
            gt_normals = np.asarray(gt_pc.normals)
            gt_pc = torch.cat([torch.tensor(gt_xyz, dtype=torch.float32, device=device), torch.tensor(gt_normals, dtype=torch.float32, device=device)], dim=1)
        else:
            gt_pc = ori_pc.clone()
            
    if not use_origin_normal:
        input_pc = util.estimate_normals(input_pc, max_nn=10)    
    else:
        input_pc = ori_pc.clone()
    
    input_pc =  propagation_method(input_pc,verbose=verbose,*args, **kwargs)
    if verbose:
        util.draw_pc(input_pc, path=Path(output_path + "/single_%s_result.ply"% propagation_method.__name__))
    # 如果有gt_pc,计算误差
    if gt_pc.shape[1] == 6:
        metrics = util.cal_metrics(gt_pc,input_pc)
        print("metrics:",metrics)
        return metrics

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
        metrics = util.cal_metrics(gt_pc,input_pc)
        print("loss:",metrics)
        return metrics

import threading

flip_times = 9


# 运行两类文件，gt和res
def run_res_and_compare(gt_path):
    head = "filename,"
    printmsg = "%s," % gt_path
    res_path = gt_path.replace("gt","res")
    MyTimer = util.timer_factory()
    head += "origin_loss,"
    
    _,ori_pc = util.load_and_trans_tensor(res_path)
    _,gt_pc = util.load_and_trans_tensor(gt_path)
    metrics = util.cal_metrics(gt_pc,ori_pc)
    printmsg += "%s," % str(metrics['count_90'] / metrics['total_count'])

    with MyTimer('xie on tree'):
        tree_xie_loss = single_propagate_file(res_path,use_origin_normal=True,propagation_method=xie_tree_propagation_points_file,times=flip_times,gt_path=gt_path,verbose=False)
        print("\n")
        head += "tree_xie_loss,"
        printmsg += "%s," % str(tree_xie_loss['count_90'] / tree_xie_loss['total_count'])
    print("\n")
    
    with MyTimer('xie dipole'):
        xie_loss = single_propagate_file(res_path,use_origin_normal=True,propagation_method=xie_propagation_points_file,gt_path=gt_path,verbose=False)
        print("\n")
        head += "xie_loss,"
        printmsg += "%s," % str(xie_loss['count_90'] / xie_loss['total_count'])
        
    with MyTimer('st dipole'):
        dipole_loss = single_propagate_file(res_path,use_origin_normal=True,propagation_method=st_propagation_points_file,gt_path=gt_path,verbose=False)
        print("\n")
        head += "dipole_loss"
        printmsg += "%s," % str(dipole_loss['count_90'] / dipole_loss['total_count'])
    
    return printmsg,head
    
        

def run_file(file,verbose=False) -> str:
    head = "filename,"
    printmsg = "%s," % file
    MyTimer = util.timer_factory() 
    
    metrics = ['loss','count_90','total_count']
    # head += ",".join(metrics) + ","
    
    # with MyTimer('xie on tree with pointWeight'):
    #     gt_tree_xie_loss = single_propagate_file(file,use_origin_normal=True,propagation_method=xie_tree_propagation_points_file,times=flip_times,use_pw=True,verbose=verbose)
    #     print("\n")
    #     head += "gt_tree_xie_with_pw_loss,"
    #     printmsg += "%s," % str(gt_tree_xie_loss['count_90'] / gt_tree_xie_loss['total_count'])
        
    #     tree_xie_loss = (single_propagate_file(file,use_origin_normal=False,propagation_method=xie_tree_propagation_points_file,times=flip_times,use_pw=True,verbose=verbose))
    #     print("\n")        
    #     head += "tree_xie_with_pw_loss," 
    #     printmsg += "%s," % str(tree_xie_loss['count_90'] / tree_xie_loss['total_count'])
    # print("\n")
    

    with MyTimer('xie on tree'):
        # gt_tree_xie_loss = single_propagate_file(file,use_origin_normal=True,propagation_method=xie_tree_propagation_points_file,times=flip_times,verbose=verbose,use_pw=False)
        # head += "gt_tree_xie_loss,"
        # printmsg += "gt_tree_xie_loss,%s," % str(gt_tree_xie_loss['count_90'] / gt_tree_xie_loss['total_count'])
        # print("\n")
        tree_xie_loss = single_propagate_file(file,use_origin_normal=False,propagation_method=xie_tree_propagation_points_file,times=flip_times,verbose=verbose,use_pw=False)
        print("\n")        
        head += "tree_xie_loss," 
        printmsg += "%s," % str(tree_xie_loss['count_90'] / tree_xie_loss['total_count'])
    print("\n")
    
    # with MyTimer('xie st'):
    #     # gt_xie_loss = str(single_propagate_file(file,use_origin_normal=True,propagation_method=xie_propagation_points_file))
    #     # print("\n")
        
    #     xie_loss = (single_propagate_file(file,use_origin_normal=False,propagation_method=xie_propagation_points_file,verbose=verbose))
    #     print("\n")
    
    #     head += "xie_loss,"
    #     printmsg += "%s," % str(xie_loss['count_90'] / xie_loss['total_count'])
    # print("\n")
        
    # with MyTimer('dipole st'):
    #     # gt_dipole_loss = single_propagate_file(file,use_origin_normal=True,propagation_method=st_propagation_points_file)
    #     # head += "gt_dipole_loss,"
    #     # printmsg += "%s," % str(gt_dipole_loss['count_90'] / gt_dipole_loss['total_count'])
    #     # print("\n")
        
    #     dipole_loss = (single_propagate_file(file,use_origin_normal=False,propagation_method=st_propagation_points_file,verbose=verbose))
    #     print("\n")
    #     head += "dipole_loss"
    #     printmsg += "%s," % str(dipole_loss['count_90'] / dipole_loss['total_count'])
    return printmsg,head


def run_floder(floder,exp_name,if_parallel=False,hander=run_file):
    pc_list = os.listdir(floder)
    if os.path.exists("temp") == False:
        os.mkdir("temp")
    if os.path.exists("temp/%s.csv" % exp_name):
        print("Error: log file already exists, would you like to overwrite?")
        flag = input()
        if flag != "y":
            return
    log = open("temp/%s.csv" % exp_name,"w")
    # lock
    lock = threading.Lock()
    threads = []
    # 定义一个全局变量，用于判断head是否已经写入
    head_writed = False
    
    def single_handle(filename):
        if filename[-3:] == "ply":
            msg,head = hander(floder + filename)
            lock.acquire()
            log = open("temp/%s.csv" % exp_name,"a")
            nonlocal head_writed
            if not head_writed:
                log.write(head + "\n")
                head_writed = True
            print("=============================================")
            print(msg)
            log.write(msg + "\n")
            log.close()
            lock.release()

    if if_parallel:
        for pc in pc_list:
            if pc[-3:] != "ply" :
                continue
            # 创建线程
            t = threading.Thread(target=single_handle,args=(pc,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        for pc in pc_list:
            if pc[-3:] != "ply" :
                continue
            single_handle(pc)
    log.close()


if __name__ == '__main__':
    MyTimer = util.timer_factory()
    # run_file(input_pc_path,True)
    # verbose_run_file = lambda x: run_file(x,True)
    with MyTimer('run floder'): 
        # print(run_res_and_compare(input_pc_path))
        run_floder("D:\Documents\zhudoongli\CG\project/NormalEstimation\dipole-normal-prop/data/gt_test_2/","dipole5",hander=run_file)    