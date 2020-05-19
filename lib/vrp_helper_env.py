import numpy as np
import copy
from lib.vrp_helper_1 import get_solDist

def build_edge_index_routes(batch_routes, inverse=False):
    edge_index_routes = []
    for i_batch in range(len(batch_routes)):
        froms = []
        tos = []
        routes = batch_routes[i_batch]
        for r in routes:
            r0 = [0]+r+[0]
            if inverse:
                for i_tmp in range(len(r0)-2):
                    tos.append(r0[i_tmp])
                    froms.append(r0[i_tmp+1])
            else:
                for i_tmp in range(1,len(r0)-1):
                    froms.append(r0[i_tmp])
                    tos.append(r0[i_tmp+1])
        edge_index_routes.append(np.array([froms, tos]))
    return edge_index_routes

def build_edge_index_near(distMats, n, inverse=False):
    edge_index_routes = []
    for i_batch in range(len(distMats)):
        froms = []
        tos = []
        distMat = distMats[i_batch]
        for anchor in range(len(distMat)):
            for node in np.argsort(distMat[anchor])[1:n+1]:
                if inverse:
                    tos.append(node)
                    froms.append(anchor)
                else:
                    froms.append(node)
                    tos.append(anchor)
        edge_index_routes.append(np.array([froms, tos]))
    return edge_index_routes

def build_action_index(n_nodes):
    result = []
    for i in range(1,n_nodes+1):
        for j in range(1,n_nodes+1):
            result.append((i,j,True))
    for i in range(1,n_nodes+1):
        for j in range(1,n_nodes+1):
            result.append((i,j,False))
    return result

def find_ruins(n_ruins, anchor, coefficient, neighbour, routes, ruined):
    """
    与 SISR 规则类似。区别在于:
        1. 路线中被删去部分的比例是输入项，而非通过 l_max, c_bar 等参数随机。
        2. 如果下一个 neighbour 所在的线路被删除过，该点依然可以被删除，且一样会寻找线路前后的点删除。
    以上修改使得 coefficient 和删去点的空间聚集度具有更严格的正相关性
    """
    def find_t(routes, c):
        for i in range(len(routes)):
            if c in routes[i]: return i
        return None
    
    def remove_nodes(route, c, num):
        nodes = []
        i = route.index(c)
        nodes.append(route[i])
        j = i+1
        k = i-1
        num = min(num, len(route))
        while len(nodes)<num:
            if j>=len(route):
                j=0
            if k<0:
                k = len(route)-1
            nodes.append(route[j])
            if len(nodes)<num:
                nodes.append(route[k])
            j+=1
            k-=1
        return nodes

    coefficient = max(coefficient, 1e-8)
    ruin_nodes = []
    # ruined_t_indices = set([])
    for c in neighbour[anchor]:
        if c not in ruin_nodes and c!=0:
            t = find_t(routes, c)
            # if t in ruined_t_indices: continue
            num = int(np.ceil(len(routes[t]) * coefficient))
            newly_removed = remove_nodes(routes[t], c, num)
            for node in newly_removed:
                if node not in ruin_nodes and node not in ruined and len(ruin_nodes)<n_ruins: ruin_nodes.append(node)
            # ruined_t_indices.add(t)
            if len(ruin_nodes)>=n_ruins:
                break
    ruin_nodes = ruin_nodes[:n_ruins]
#     print(ruined, ruin_nodes)
    return ruin_nodes

class vrp_env:
    
    def __init__(self, cap, data, distMat, init_routes, init_dist, init_T, alpha_T, sim_anl=True):
        self.cap = cap
        self.data = data
        self.distMat = distMat
        self.curr_routes = init_routes
        self.curr_dist = init_dist
        self.best_routes = copy.deepcopy(init_routes)
        self.best_dist = init_dist
        self.T = init_T
        self.alpha_T = alpha_T
        self.sim_anl = sim_anl
        
    def get_embedding(self):
        result = []
        norm = self.data[0,4]-self.data[0,3]
        result.append(self.data[:,0]/norm) # x normalized by depot's 时间窗跨度
        result.append(self.data[:,1]/norm) # y normalized by depot's 时间窗跨度
        result.append(self.data[:,2]/self.cap)
        result.append(self.data[:,3]/norm) # start_time normalized by depot's 时间窗跨度
        result.append(self.data[:,4]/norm) # end_time normalized by depot's 时间窗跨度
        result.append(self.data[:,5]/norm) # service_time normalized by depot's 时间窗跨度
        result.append((self.data[:,4]-self.data[:,3])/norm) # 时间窗跨度除以 depot's 时间窗跨度
        result.append(np.zeros(len(self.data))) # cumulative demand normalized by cap
        result.append(np.zeros(len(self.data))) # cumulative distance normalized by depot's 时间窗跨度
        result.append(np.zeros(len(self.data))) # total demand normalized by cap
        result.append(np.zeros(len(self.data))) # total distance normalized by depot's 时间窗跨度
        for r in self.curr_routes:
            complete_r = [0]+r+[0]
            cum_demand = 0
            cum_dist = 0
            for i in range(1, len(complete_r)-1):
                cum_demand+=self.data[complete_r[i],2]
                cum_dist+=self.distMat[complete_r[i-1],complete_r[i]]
                result[7][complete_r[i]] = cum_demand
                result[8][complete_r[i]] = cum_dist
            for i in range(1, len(complete_r)-1):
                result[9][complete_r[i]] = cum_demand
                result[10][complete_r[i]] = cum_dist

        result[7]/=self.cap
        result[8]/=norm
        result[9]/=self.cap
        result[10]/=norm
        return np.transpose(np.array(result))
    
    def reward_func(self, old_d, new_d):
        # return max(-1, old_d-new_d)
        return old_d-new_d
    
    def sim_annealing(self, new_dist, new_routes):
        if new_dist<(self.curr_dist-self.T*np.log(np.random.random())):
            if new_dist<self.best_dist:
                self.best_dist = new_dist
                self.best_routes = copy.deepcopy(new_routes)
            self.curr_dist = new_dist
            self.curr_routes = new_routes

    def step(self, action):
        def find_ij(routes, node):
            for i,r in enumerate(routes):
                for j,c in enumerate(r):
                    if c==node:
                        return i,j
            return -1,-1

        def check(vehicle_capcity, data, distance_matrix, r):
            R = [0]+r+[0]
            t = 0
            demands = np.sum([data[x,2] for x in R])
            if demands>vehicle_capcity: return False
            for i in range(len(R)):
                if i==0:
                    arrive_t = t
                else:
                    arrive_t = t+distance_matrix[R[i-1],R[i]]
                start_t = max(data[R[i],3], arrive_t)
                due_t = data[R[i],4]
                serve_t = data[R[i],5]
                end_t = start_t+serve_t
                t=end_t
                if due_t<start_t: return False
            return True

        def get_route_distance(distance_matrix, route):
            r = [0]+route+[0]
            result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
            return result

        node1, node2, isPrevious = action
        ij1 = find_ij(self.curr_routes, node1)
        ij2 = find_ij(self.curr_routes, node2)
        if ij1[0]!=ij2[0]:
            r2 = copy.deepcopy(self.curr_routes[ij2[0]])+[0]
            if isPrevious:
                r2 = r2[:ij2[1]]+[node1]+r2[ij2[1]:]
            else:
                r2 = r2[:(ij2[1]+1)]+[node1]+r2[(ij2[1]+1):]
            r2 = r2[:-1]
            if check(self.cap, self.data, self.distMat, r2):
                new_routes = copy.deepcopy(self.curr_routes)
                new_routes[ij1[0]] = new_routes[ij1[0]][:ij1[1]]+new_routes[ij1[0]][(ij1[1]+1):]
                new_routes[ij2[0]] = r2
                dist = get_solDist(self.distMat, new_routes)
                r = self.reward_func(self.curr_dist, dist)
                if self.sim_anl:
                    self.sim_annealing(dist, new_routes)
                else:
                    self.curr_routes = new_routes
                    self.curr_dist = dist
                self.T *= self.alpha_T
            else:
                r = -1.0
        else:
            r1 = copy.deepcopy(self.curr_routes[ij2[0]])+[0]
            r1 = r1[:ij1[1]]+r1[(ij1[1]+1):]
            try:
                j2 = r1.index(node2)
            except ValueError:
                print(r1, action, self.curr_routes)
                raise
            if isPrevious:
                r1 = r1[:j2]+[node1]+r1[j2:]
            else:
                r1 = r1[:(j2+1)]+[node1]+r1[(j2+1):]
            r1 = r1[:-1]
            if check(self.cap, self.data, self.distMat, r1):
                new_routes = copy.deepcopy(self.curr_routes)
                new_routes[ij1[0]] = r1
                dist = get_solDist(self.distMat, new_routes)
                r = self.reward_func(self.curr_dist, dist)
                if self.sim_anl:
                    self.sim_annealing(dist, new_routes)
                else:
                    self.curr_routes = new_routes
                    self.curr_dist = dist
                self.T *= self.alpha_T
            else:
                r = -1.0

        return r
    
class vrp_env_sisr:
    
    def __init__(self, cap, data, distMat, init_routes, init_dist, init_T, alpha_T, sim_anl=True):
        self.cap = cap
        self.data = data
        self.distMat = distMat
        self.curr_routes = init_routes
        self.curr_dist = init_dist
        self.best_routes = copy.deepcopy(init_routes)
        self.best_dist = init_dist
        self.neighbours = self.get_neighbours()
        self.T = init_T
        self.alpha_T = alpha_T
        self.sim_anl = sim_anl
        
    def get_embedding(self):
        result = []
        norm = self.data[0,4]-self.data[0,3]
        result.append(self.data[:,0]/norm) # x normalized by depot's 时间窗跨度
        result.append(self.data[:,1]/norm) # y normalized by depot's 时间窗跨度
        result.append(self.data[:,2]/self.cap)
        result.append(self.data[:,3]/norm) # start_time normalized by depot's 时间窗跨度
        result.append(self.data[:,4]/norm) # end_time normalized by depot's 时间窗跨度
        result.append(self.data[:,5]/norm) # service_time normalized by depot's 时间窗跨度
        result.append((self.data[:,4]-self.data[:,3])/norm) # 时间窗跨度除以 depot's 时间窗跨度
        result.append(np.zeros(len(self.data))) # cumulative demand normalized by cap
        result.append(np.zeros(len(self.data))) # cumulative distance normalized by depot's 时间窗跨度
        result.append(np.zeros(len(self.data))) # total demand normalized by cap
        result.append(np.zeros(len(self.data))) # total distance normalized by depot's 时间窗跨度
        for r in self.curr_routes:
            complete_r = [0]+r+[0]
            cum_demand = 0
            cum_dist = 0
            for i in range(1, len(complete_r)-1):
                cum_demand+=self.data[complete_r[i],2]
                cum_dist+=self.distMat[complete_r[i-1],complete_r[i]]
                result[7][complete_r[i]] = cum_demand
                result[8][complete_r[i]] = cum_dist
            for i in range(1, len(complete_r)-1):
                result[9][complete_r[i]] = cum_demand
                result[10][complete_r[i]] = cum_dist

        result[7]/=self.cap
        result[8]/=norm
        result[9]/=self.cap
        result[10]/=norm
        return np.transpose(np.array(result))
    
    def get_neighbours(self):
        n_vertices = self.distMat.shape[0]
        neighbours = []
        for i in range(n_vertices):
            index_dist = [(j, self.distMat[i][j]) for j in range(n_vertices)]
            sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
            neighbours.append([x[0] for x in sorted_index_dist])
        return neighbours
    
    def get_ruins(self, anchor, coefficient, n_ruins, ruined=[]):
        return find_ruins(n_ruins, anchor, coefficient, self.neighbours, self.curr_routes, ruined)
    
    def reward_func(self, old_d, new_d):
        # return max(-1, old_d-new_d)
        return old_d-new_d
    
    def sim_annealing(self, new_dist, new_routes):
        b = False
        reward = self.reward_func(self.curr_dist, new_dist)
        if new_dist<(self.curr_dist-self.T*np.log(np.random.random())):
            b = True
            if new_dist<self.best_dist:
                self.best_dist = new_dist
                self.best_routes = copy.deepcopy(new_routes)
            self.curr_dist = new_dist
            self.curr_routes = new_routes
        self.T *= self.alpha_T
        return b, reward