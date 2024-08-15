import socket
import numpy as np
import json
from options import get_parser
import options
from pathlib import Path
from field_utils import *
torch.manual_seed(1)

# 设置服务器的IP和端口
HOST = '0.0.0.0'  # 监听所有IP地址
PORT = 12345      # 监听的端口号
REQUEST_BUFFER_SIZE = 1000 # 接收缓冲区大小，单位为字节

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))


def simple_estimate(xyz_data,config):
    input_pc = util.npxyz2tensor(xyz_data).to(device)
    input_pc = util.estimate_normals(input_pc, max_nn=30)
    input_pc, transform = util.Transform.trans(input_pc)
    strongest_field_propagation_points(input_pc, diffuse=config['diffuse'], starting_point=0)
    if measure_mean_potential(input_pc) < 0:
        input_pc[:, 3:] *= -1
    transformed_pc = transform.inverse(input_pc).transpose(0, 1)
    return transformed_pc

# def hoppe_estimate(xyz_data,config):
    


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")

                # 接收数据
                req = conn.recv(REQUEST_BUFFER_SIZE)
                req = json.loads(req.decode())
                print(req)
                data_buffer_size = req['data_size'] * 24 # 24 bytes per point
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
                    break
                print(f"Received {len(data)} bytes in total")
                if len(data) != data_buffer_size:
                    print(f"Data size mismatch. Expected {data_buffer_size} bytes, but received {len(data)} bytes.")
                    assert False
                # 假设接收到的数据是二进制形式的XYZ浮点数数组
                xyz_data = np.frombuffer(data, dtype=np.float64).reshape(-1, 3)

                # 计算法向量
                if req['function_name'] == 'simple_estimate':
                    transformed_pc = simple_estimate(xyz_data, req['function_config'])
                    result = transformed_pc
                else:
                    print(f"Unknown method: {req['function_name']}")
                    assert False

                # 返回结果
                conn.sendall(result.astype(np.float64).tobytes())
                conn.close()

if __name__ == "__main__":
    main()
