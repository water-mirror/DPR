import numpy as np
import math

def random_cvrp(n_nodes, n_clusters=None, demand_lowerBnd=1, demand_upperBnd=9):
    data = []
    # 如果 node 数量小于1000，那么边长为1000
    side_limit = int(max(100.01, n_nodes/10))
    
    if n_clusters is not None:
        assert n_clusters<n_nodes
        while len(data) < n_clusters:
            coord = [np.random.randint(0,side_limit), np.random.randint(0,side_limit)]
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],
                         np.random.randint(demand_lowerBnd, demand_upperBnd+1),])
        
        while len(data) < n_nodes:
            rnd = np.array([np.random.randint(-3,4), np.random.randint(-3,4)])
            coord = data[np.random.randint(len(data))][:2]+rnd
            if coord[0]<0 or coord[1]<0 or coord[0]>=side_limit or coord[1]>=side_limit: continue
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],
                         np.random.randint(demand_lowerBnd, demand_upperBnd+1),])
    else:
        while len(data) < n_nodes:
            coord = [np.random.randint(0,side_limit), np.random.randint(0,side_limit)]
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],
                         np.random.randint(demand_lowerBnd, demand_upperBnd+1),])
    data = np.array(data)
    return data

def random_vrptw(n_nodes, n_clusters=None, demand_lowerBnd=1, demand_upperBnd=9):
    
    DEPOT_END = 300
    SERVICE_TIME = 10
    TW_WIDTH = 30
    
    def get_distance(vec1, vec2):
        return np.sum((vec1-vec2)**2)**0.5

    def random_tw(dist_to_depot,service_time,depot_end,tw_width):
        _tmp_0 = math.ceil(dist_to_depot)
        # _tmp_1 = int((_tmp_0+depot_end)/2)
        _tmp_1 = 200
        start = np.random.randint(_tmp_0, _tmp_1)
        end = start + tw_width
        if end < dist_to_depot or end + service_time + dist_to_depot > depot_end:
            start = 0
            end = depot_end
        return start,end
    
    data = random_cvrp(n_nodes, n_clusters, demand_lowerBnd, demand_upperBnd)
    tw = [[0,DEPOT_END,0]]
    for i in range(1, len(data)):
        dist_to_depot = get_distance(data[0], data[i])
        start,end = random_tw(dist_to_depot,SERVICE_TIME,DEPOT_END,TW_WIDTH)
        tw.append([start,end,SERVICE_TIME])
    tw = np.array(tw)
    result = np.concatenate((data, tw), axis=1)
    return result