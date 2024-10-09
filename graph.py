import field_utils
import util
import numpy as np
import gurobipy as gp

import numpy as np
'''
    Parameters:
    - mesh: A mesh represented as a tuple (vertices, faces), where vertices is a list of 3D points and faces is a list of vertex indices for each face
    - topology: A mesh represented as a tuple (vertices, faces) to add to the mesh
    Returns:
    - None
'''
def add_topology(mesh,topology):
    start = len(mesh[0])
    for i in topology[0]:
        mesh[0].append(i)
    for i in topology[1]:
        t = []
        for j in i:
            t.append(j + start)
        mesh[1].append(t)
    return 0

def normalize(v):
    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    v[0] /= norm
    v[1] /= norm
    v[2] /= norm

'''
    Parameters:
    - start: The starting point of the arrow
    - end: The ending point of the arrow
    - radius: The radius of the arrow
    - n: the longtitude precision of the arrow
    Returns:
    - A mesh representing the arrow
'''
def get_arrow(start, end, radius = 0.001, n = 10):
    rate = 2
    dir = end - start
    z = dir.copy()
    normalize(z)
    x = np.array([1, 0, 0])
    if np.linalg.norm(x - z) < 0.01:
        x = np.array([0, 1, 0])
    y = np.cross(z, x)
    normalize(y)
    x = np.cross(y, z)
    normalize(x)

    end = start + dir * 0.95
    start = start + dir * 0.05
    dir = end - start

    cylinder_end = start + dir * 0.9

    res = ([], [])
    for i in range(n):
        theta = 2 * np.pi * i / n
        p = start + radius / rate * (np.cos(theta) * x + np.sin(theta) * y)
        res[0].append(p)

    for i in range(n):
        theta = 2 * np.pi * i / n
        p = cylinder_end + radius / rate * (np.cos(theta) * x + np.sin(theta) * y)
        res[0].append(p)

    for i in range(n):
        res[1].append([n + i, i, (i + 1) % n])
        res[1].append([n + i, (i + 1) % n, (i + 1) % n + n])
    cend = len(res[0])

    for i in range(n):
        theta = 2 * np.pi * i / n
        p = cylinder_end + radius * (np.cos(theta) * x + np.sin(theta) * y)
        res[0].append(p)

    res[0].append(end)
    top = len(res[0]) - 1

    for i in range(n):
        res[1].append([top, i + cend, (i + 1) % n + cend])
    return res

'''
    Parameters:
    - center: The center of the sphere
    - radius: The radius of the sphere
    - n: The latitude precision of the sphere
    - m: The longitude precision of the sphere 
    Returns:
    - A mesh representing the sphere
'''
def get_sphere(center, radius = 0.008, n = 10, m = 10):
    res = ([], [])
    for i in range(n):
        for j in range(m):
            theta = 2 * np.pi * i / n
            phi = np.pi * j / m
            p = center + np.array([radius * np.sin(phi) * np.cos(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(phi)])
            res[0].append(p)
    for i in range(n):
        for j in range(m):
            a = i * m + j
            b = i * m + (j + 1) % m
            c = ((i + 1) % n) * m + j
            d = ((i + 1) % n) * m + (j + 1) % m
            res[1].append([a, b, c])
            res[1].append([b, d, c])
    return res


class BiEdge:
    def __init__(self,u,v,w,invw):
        self.u = u
        self.v = v
        self.w = w
        self.invw = invw
    
    def __iter__(self):
        return iter([self.u,self.v,self.w,self.invw])

class BidGraph:
    def __init__(self):
        self.V = [] # id list
        self.E = [] # BiEdge list
              
    def to_matrix(self):
        # 检验nodes的id是否是连续的
        set_id = set(self.V)
        for i in range(len(set_id)):
            if not i in set_id:
                print("Error! the graph's vertex id is not continuous")
                assert i in set_id
        
        n = len(self.V)
        A = np.zeros((n,n))
        B = np.zeros((n,n))
        for edg in self.E:
            A[edg.u][edg.v] = edg.w
            A[edg.v][edg.u] = edg.w
            B[edg.u][edg.v] = edg.invw
            B[edg.v][edg.u] = edg.invw
        return A,B

    def add_edge(self,u,v,calculator):
        self.V.append(u)
        self.V.append(v)
        w,invw = calculator(u,v)
        self.E.append(BiEdge(u,v,w,invw))
        return self


'''
计算一个指派的weight_sum
x: 一个指派, 一个n维的向量,取值为0或1
A: 一个n*n的矩阵, A[i,j]表示x[i]和x[j]在指派相同的时候的权重
B: 一个n*n的矩阵, B[i,j]表示x[i]和x[j]在指派不同的时候的权重
return: weight_sum
'''
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

def MIQP(A,B):
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]
    # Create a new model
    m = gp.Model("mip1")
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



