import field_utils
import socket
import numpy as np
import json
import torch
import util
import threading
torch.manual_seed(1)

# 设置服务器的IP和端口
HOST = '0.0.0.0'  # 监听所有IP地址
PORT = 12345     # 监听的端口号
REQUEST_BUFFER_SIZE = 1000 # 接收缓冲区大小，单位为字节
max_thread = 50 # 同时处理的最大线程数

max_pts = 30000 * 30000 # 同时处理的最大点数
res_pts = max_pts # 当前剩余点数
pts_mtx = threading.Lock()     
    
device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

def log_msg(msg,mode='error'):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if mode == 'error':
        print("\033[1;31m" + msg + "\033[0m")
    elif mode == 'warning':
        print("\033[1;33m" + msg + "\033[0m")
    else:
        print("\033[1;32m" + msg + "\033[0m")
        
    log_path = mode + ".log"
    with open(log_path, 'a') as f:
        f.write(current_time + ' ' + msg + '\n')
    return

# 申请处理点资源，如果点数超过最大值，则等待
def apply_pts(len_pts):
    sum_pts = len_pts * len_pts
    global res_pts
    global pts_mtx
    global max_pts
    if sum_pts > max_pts:
        msg = f"Too many points to process at once. {sum_pts} points requested, but only {max_pts} points allowed."
        log_msg(msg,mode='error')
        print(msg)
        assert False

    pts_mtx.acquire()
    flag = True
    while sum_pts > res_pts:
        pts_mtx.release()
        if flag:
            print(f"Waiting for points. Apllying {sum_pts} points, but only {res_pts} points left.")
            flag = False
        time.sleep(20)
        pts_mtx.acquire()
    res_pts -= sum_pts
    pts_mtx.release()
    return    

def release_pts(len_pts):
    global res_pts
    global pts_mtx
    pts_mtx.acquire()
    res_pts += len_pts * len_pts
    pts_mtx.release()
    return


def tree_xie_propagation(xyz_data,config):
    if len(xyz_data) < config['max_nn']:
        log_msg(f"Point number is less than max_nn. {len(xyz_data)} points requested, but at least {config['max_nn']} points required.",mode='warning')
        xyz_normal = np.ones((len(xyz_data), 6))
        xyz_normal[:, :3] = xyz_data
        return xyz_normal
    input_pc = util.npxyz2tensor(xyz_data).to(device)
    input_pc = util.estimate_normals(input_pc, max_nn=config['max_nn'])
    input_pc, transform = util.Transform.trans(input_pc)
    field_utils.xie_propagation_points_onbfstree(input_pc, eps=config['eps'], diffuse=config['diffuse'], times=config['times'], starting_point=0,knn_mask=config['knn_mask'],C=config['C'])
    if field_utils.measure_mean_potential(input_pc) < 0:
        input_pc[:, 3:] *= -1
    transformed_pc = transform.inverse(input_pc)
    transformed_pc = transformed_pc.cpu().numpy()
    return transformed_pc

def simple_estimate(xyz_data,config):
    input_pc = util.npxyz2tensor(xyz_data).to(device)
    input_pc = util.estimate_normals(input_pc, max_nn=30)
    # util.draw_pc(input_pc, path=Path("data/output/server_init.ply"))
    input_pc, transform = util.Transform.trans(input_pc)
    field_utils.strongest_field_propagation_points(input_pc, diffuse=config['diffuse'], starting_point=0)
    if field_utils.measure_mean_potential(input_pc) < 0:
        input_pc[:, 3:] *= -1
    transformed_pc = transform.inverse(input_pc)
    transformed_pc = transformed_pc.cpu().numpy()
    # util.draw_pc(transformed_pc, path=Path("data/output/server_result.ply"))
    return transformed_pc

def xie_propagation(xyz_data,config):
    input_pc = util.npxyz2tensor(xyz_data).to(device)
    input_pc = util.estimate_normals(input_pc, max_nn=config['max_nn'])
    input_pc, transform = util.Transform.trans(input_pc)
    field_utils.xie_propagation_points(input_pc, eps=config['eps'], diffuse=config['diffuse'], starting_point=0)
    if field_utils.measure_mean_potential(input_pc) < 0:
        input_pc[:, 3:] *= -1
    transformed_pc = transform.inverse(input_pc)
    transformed_pc = transformed_pc.cpu().numpy()
    return transformed_pc

import graph_dipole
def graph_dipole_estimate(xyz_data,config):
    return graph_dipole.graph_dipole_server_api(xyz_data,config)

def hoppe_estimate(xyz_data,config):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_data)
    r = 0.1
    if 'r' in config:
        r = config['r']
    k_neighbor = 10
    if 'k_neighbor' in config:
        k_neighbor = config['k_neighbor']
    _lambda = 0.1
    if 'lambda' in config:
        _lambda = config['lambda']
    _alpha = 0.5
    if 'alpha' in config:
        _alpha = config['alpha']

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=k_neighbor))
    pcd.orient_normals_consistent_tangent_plane(k_neighbor, _lambda, _alpha)
    normals = np.asarray(pcd.normals)
    res = np.concatenate([xyz_data, normals], axis=1)
    return res

def handle_client(conn, addr):
    with conn:
        print(f"Connected by {addr}")
        try:
            # 接收数据
            req = conn.recv(REQUEST_BUFFER_SIZE)
            req = json.loads(req.decode())
            print(req)
            data_buffer_size = req['data_size'] * 24
            repo = json.dumps({"status": "OK"})
            conn.sendall(repo.encode())
            data_recv = 0
            data = b''
            while data_recv < data_buffer_size:
                tdata = conn.recv(data_buffer_size - data_recv)
                data_recv += len(tdata)
                if not tdata:
                    break
                print(f"Received {len(tdata)} bytes")
                data += tdata
            if not data:
                return
            print(f"Received {len(data)} bytes in total")
            if len(data) != data_buffer_size:
                print(f"Data size mismatch. Expected {data_buffer_size} bytes, but received {len(data)} bytes.")
                assert False
            # 假设接收到的数据是二进制形式的XYZ浮点数数组
            xyz_data = np.frombuffer(data, dtype=np.float64).reshape(-1, 3)
            apply_pts(len(xyz_data))
            # 计算法向量
            if req['function_name'] == 'simple_estimate':
                transformed_pc = simple_estimate(xyz_data, req['function_config'])
                result = transformed_pc
            elif req['function_name'] == 'hoppe_estimate':
                transformed_pc = hoppe_estimate(xyz_data, req['function_config'])
                result = transformed_pc
            elif req['function_name'] == 'graph_dipole_estimate':
                transformed_pc = graph_dipole_estimate(xyz_data, req['function_config'])
                result = transformed_pc
            elif req['function_name'] == 'xie_propagation':
                transformed_pc = xie_propagation(xyz_data, req['function_config'])
                result = transformed_pc
            elif req['function_name'] == 'tree_xie_propagation':
                transformed_pc = tree_xie_propagation(xyz_data, req['function_config'])
                result = transformed_pc
            else:
                print(f"Unknown method: {req['function_name']}")
                log_msg(f"Unknown method: {req['function_name']}",mode='error')
                assert False

            # 返回结果
            conn.sendall(result.astype(np.float64).tobytes())
            release_pts(len(xyz_data))
        except Exception as e:
            print(f"Error: {e}")
            log_msg(f"Error: {e}",mode='error')
            conn.sendall(json.dumps({"status": "ERROR"}).encode())
        finally:
            conn.close()
    
import threading
import time

def multithread():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")        
        while True:
            conn, addr = s.accept()
            while threading.active_count() > max_thread:
                time.sleep(1)
            t = threading.Thread(target=handle_client, args=(conn, addr))
            t.start()
            print(f"Active threads: {threading.active_count()}")

def single_thread():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)
            print(f"Active threads: {threading.active_count()}")
     
import argparse
if __name__ == "__main__":
    # 输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=PORT, help='Port number')
    parser.add_argument('--max_thread', type=int, default=max_thread, help='Max thread number')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    PORT = args.port
    max_thread = args.max_thread
    if max_thread > 1:
        multithread()
    else:
        single_thread()
