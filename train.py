import os
import sys
import shutil
import copy
import time
import pickle
# import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# from lib.vrp_helper_legacy import random_vrptw
from lib.vrp_helper_0 import random_vrptw
from lib.vrp_helper_0 import check_vrp_route_validity, check_vrp_routes_validity
from lib.vrp_helper_1 import get_distMat, get_inbalance_distMat, get_solDist
from lib.vrp_helper_1 import parse_vrp_question, parse_vrp_answer, getQADict
from lib.vrp_helper_env import build_edge_index_routes, build_edge_index_near, vrp_env_sisr
from lib.ppo_1 import Memory, Agent

import os.path
import subprocess
from subprocess import STDOUT,PIPE

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data, DataLoader

from arguments import args
import argparse

############################################################
# 与 Java 交换数据相关的函数

def write_data_file(data, path="data.txt"):
    lines = []
    for i in range(data.shape[0]):
        lines.append(" ".join([str(x) for x in data[i]])+"\n")
    with open(path, "w+") as f:
        for line in lines:
            f.write(line)

def write_distMat_file(dist_matrices, path):

    def get_dist_matrix_str(dist_matrix):
        dist_matrix_str = []
        for i in range(len(dist_matrix)):
            dist_matrix_str.append(",".join([str(x) for x in np.round(dist_matrix[i],4)]))
        result = ";".join(dist_matrix_str)
        return result

    lines = []
    for i in range(len(dist_matrices)):
        lines.append(get_dist_matrix_str(dist_matrices[i])+"\n")
    with open(path, "w+") as f:
        for line in lines:
            f.write(line)

def write_states_file(batch_caps, batch_ruins, batch_routes, path):
    lines = []
    for i in range(len(batch_caps)):
        content = [str(batch_caps[i]), ",".join([str(x) for x in batch_ruins[i]])]
        routes_str_list = []
        for r in batch_routes[i]:
            routes_str_list.append(",".join([str(x) for x in r]))
        content.append(";".join(routes_str_list))
        lines.append(":".join(content)+"\n")
    with open(path, "w+") as f:
        for line in lines:
            f.write(line)

def compile_java(java_file):
    subprocess.check_call(['javac', java_file])

def execute_java(java_file, stdin):
    java_class,ext = os.path.splitext(java_file)
    cmd = ['java', java_class]+stdin
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    stdout,stderr = proc.communicate("".encode())
    return stdout.decode()

def parse_output(_output):
    dist, routes = _output.split(":")
    dist = float(dist)
    routes = [[int(y) for y in x.split(",")] for x in routes.split(";")]
    return dist, routes

def vrp_java(java_file, stdin):

    def compile_java(java_file):
        subprocess.check_call(['javac', java_file])

    def execute_java(java_file, stdin):
        java_class,ext = os.path.splitext(java_file)
        cmd = ['java', java_class]+stdin
        proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        stdout,stderr = proc.communicate("".encode())
        return stdout.decode()

    def parse_output(_output):
        dist, routes = _output.split(":")
        dist = float(dist)
        routes = [[int(y) for y in x.split(",")] for x in routes.split(";")]
        return dist, routes

    compile_java(java_file)
    _output = execute_java(java_file, stdin)
    return parse_output(_output)

if __name__ == "__main__":
    args = args()

    device = args.device
    starting_iter = 0
    n_iter = args.n_iter
    n_iter_max = n_iter
    n_batch = args.batch
    n_benchs = args.benchs
    n_epoch = args.epoch
    n_anchors = args.anchors

    lr = args.lr
    betas = (0.9, 0.999)
    K_epochs = args.k_epoch
    eps_clip = args.eps_clip
    train_batch_size = n_batch

    gamma = args.gamma
    embedding_dim = 11
    node_dim = args.node_dim
    critic_dims = args.critic_dims

    # 模拟退火
    init_T = args.init_T
    final_T = args.final_T
    alpha_T = (final_T/init_T)**(1.0/n_iter)

    model_folder = './models/ppo_1'
    save_path = model_folder+'/ppo.pth'

    data_root = "tmp_datas"
    state_path = "states.txt"
    distMat_path = "distMat.txt"
    try:
        os.mkdir(data_root)
    except FileExistsError:
        shutil.rmtree(data_root)
        os.mkdir(data_root)

    java_solver_rand = 'java_solverRAND.java'
    java_solver_sisr2 = 'java_solverSISR_revised.java'
    java_steper = 'java_recreate.java'

    try:
        os.mkdir(model_folder)
        print("Model folder created.")
    except FileExistsError:
        print("Model folder already exists.")
    _=[os.remove(filename) for filename in os.listdir() if ".class" in filename] # 清空目录下的 Java 编译文件

    compile_java(java_solver_rand)
    compile_java(java_solver_sisr2)
    compile_java(java_steper)
    memory = Memory()

    ###############################################################################################

    QA200 = getQADict("data/homberger_200_customer_instances", "data/solutions_200")
    q200_paths = sorted(list(QA200.keys()))
    q200 = [parse_vrp_question(q) for q in q200_paths]
    solomon_paths = ["data/solomon/"+x for x in os.listdir("data/solomon") if x[0]!='.']
    q100 = [parse_vrp_question(q) for q in solomon_paths]
    q50 = [(x[0],x[1][:51]) for x in q100]
    q25 = [(x[0],x[1][:26]) for x in q100]
    cheat_ques = [q200,q100,q50,q25]

    ###############################################################################################

    # torch.autograd.set_detect_anomaly(True)
    agent = Agent(lr, betas, gamma, K_epochs, eps_clip, train_batch_size,
                  embedding_dim, node_dim, critic_dims, n_anchors, device=device)
    agent.load(save_path)

    train_start_time = time.time()
    for i_epoch in range(n_epoch):
        n_nodes = [200,100,50,25][i_epoch%4]
        n_ruins_d = max(n_nodes**0.5, 1)*1.2
        n_ruins = int(np.ceil(n_ruins_d/n_anchors))
        n_ruins_total = int(n_ruins*n_anchors)
        print("Epoch:", i_epoch, "-- # nodes =", n_nodes)
        cheat = np.random.random()<0.3
        if cheat:
            problems = [cheat_ques[i_epoch%4][i] for i in np.random.choice(55, n_batch)]

        ######################################################################
        # 生成问题
        envs = []
        for i_batch in range(n_batch):
            if cheat:
                cap, data = problems[i_batch]
            else:
                cap, data = random_vrptw(n_nodes+1)
            distMat = get_distMat(data)

            data_path = data_root+"/data_"+str(i_batch)+".txt"
            write_data_file(data, path=data_path)

            args = [os.path.abspath(os.getcwd())+"/"+data_path, str(cap),
                    str(starting_iter), str(n_ruins_total)]
            _output = execute_java(java_solver_rand, args)
            dist, routes = parse_output(_output)
            env = vrp_env_sisr(cap, data, distMat, routes, dist, init_T, alpha_T)
            envs.append(env)

        write_distMat_file([env.distMat for env in envs], path=distMat_path)

        edge_index_n = build_edge_index_near([env.distMat for env in envs], 10)

        ######################################################################
        # 做 benchmark
        benchmarks_rand = []
        benchmarks_sisr2 = []
        for i_batch in range(min(n_batch, n_benchs)):
            # 使用java求解最多前 n_benchs 个 benchmark (节约时间)
            data_path = data_root+"/data_"+str(i_batch)+".txt"
            args = [os.path.abspath(os.getcwd())+"/"+data_path,
                    str(envs[i_batch].cap),
                    str(starting_iter+n_iter), str(n_ruins_total)]
            _output = execute_java(java_solver_rand, args)
            dist, routes = parse_output(_output)
            benchmarks_rand.append(dist)

        write_states_file([env.cap for env in envs[:n_benchs]],
                          [[] for _ in range(len(envs[:n_benchs]))],
                          [env.curr_routes for env in envs[:n_benchs]],
                          state_path)
        args = [os.path.abspath(os.getcwd())+"/"+data_root+"/",
                os.path.abspath(os.getcwd())+"/"+state_path,
                os.path.abspath(os.getcwd())+"/"+distMat_path,
                str(n_iter), str(n_ruins), str(n_anchors)]
        _outputs = execute_java(java_solver_sisr2, args)
        for _output in [x for x in _outputs.split("\n") if len(x)>0][:n_benchs]:
            dist, routes = parse_output(_output)
            benchmarks_sisr2.append(dist)

        init_distance = np.mean([env.curr_dist for env in envs[:n_benchs]])
        print("Mean init distance =", init_distance)
        print("Mean RAND operator =", np.mean(benchmarks_rand))
        print("Mean SISR operator =", np.mean(benchmarks_sisr2))

        time.sleep(1)
        ######################################################################
        # 开始迭代
        for i_iter in trange(n_iter):
            start_time = time.time()
            edge_index_r0 = build_edge_index_routes([env.curr_routes for env in envs], inverse=False)
            edge_index_r1 = build_edge_index_routes([env.curr_routes for env in envs], inverse=True)

            _es = [env.get_embedding() for env in envs]
            _es = np.array(_es)
            if i_iter == 0:
                input_ = np.zeros([n_batch,n_ruins_total,embedding_dim]).astype(np.float32)
                input_ = torch.from_numpy(input_).to(torch.device(device))
                h = np.zeros([n_batch,node_dim]).astype(np.float32)
                h = torch.from_numpy(h).to(torch.device(device))
            else:
                input_ = (np.array([_es[i][ruins[i]] for i in range(n_batch)])* bs).astype(np.float32)
                input_ = torch.from_numpy(input_).to(torch.device(device))
                h = h_
            time1 = time.time()-start_time

            start_time = time.time()
            data_list = []
            for i_batch in range(n_batch):
                data = Data(x=torch.from_numpy(_es[i_batch]).float().to(torch.device(device)),
                            edge_index_r0=torch.from_numpy(edge_index_r0[i_batch]).to(torch.device(device)),
                            edge_index_r1=torch.from_numpy(edge_index_r1[i_batch]).to(torch.device(device)),
                            edge_index_n=torch.from_numpy(edge_index_n[i_batch]).to(torch.device(device)),
                            state_=h[i_batch].detach().to(torch.device(device)),
                            input_=input_[i_batch].detach().to(torch.device(device)))
                data_list.append(data)
            loader = DataLoader(data_list,batch_size=n_batch,shuffle=False)
            time2 = time.time()-start_time

            start_time = time.time()
            tmp = agent.policy_old.act(list(loader)[0], n_nodes+1)
            h_, action_node, action_sisr, dist_p, dist_s = tmp

            indices = torch.arange(action_node.size(0)).repeat(n_anchors)
            indices = indices.reshape([-1,action_node.size(0)]).transpose(0,1)
            coeff = action_sisr[indices, action_node].cpu().detach().numpy()
            sisr_logs = dist_s.log_prob(action_sisr)[indices, action_node].detach()
            prob_logs = [dist_p.log_prob(action_node[:,i]).cpu().detach().numpy() for i in range(n_anchors)]
            prob_logs = torch.Tensor(prob_logs).to(torch.device(device))
            prob_logs = prob_logs.transpose(0,1)
            log_probs = prob_logs + sisr_logs
            log_probs = torch.sum(log_probs, 1)
            action_node = action_node.detach()
            time3 = time.time()-start_time

            start_time = time.time()
            ruins = []
            for i_batch in range(n_batch):
                tmp_ruins = []
                for i_a in range(n_anchors):
                    tmp = envs[i_batch].get_ruins(action_node[i_batch][i_a]+1, coeff[i_batch][i_a],
                                                  n_ruins, ruined=tmp_ruins)
                    tmp_ruins.extend(tmp)
                ruins.append(tmp_ruins)
            time4 = time.time()-start_time

            start_time = time.time()
            write_states_file([env.cap for env in envs], ruins, [env.curr_routes for env in envs], state_path)
            new_routes = []
            new_dists = []
            args = [os.path.abspath(os.getcwd())+"/"+data_root+"/",
                    os.path.abspath(os.getcwd())+"/"+state_path,
                    os.path.abspath(os.getcwd())+"/"+distMat_path]
            _outputs = execute_java(java_steper, args)
            for _output in [x for x in _outputs.split("\n") if len(x)>0]:
                dist, routes = parse_output(_output)
                new_routes.append(routes)
                new_dists.append(dist)
            time5 = time.time()-start_time

            start_time = time.time()
            bs = []
            rewards = []
            for i_batch in range(n_batch):
                b, reward = envs[i_batch].sim_annealing(new_dists[i_batch], new_routes[i_batch])
                bs.append(float(b))
                rewards.append(reward)
            bs = np.array(bs).reshape([-1,1,1])
            rewards = np.array(rewards)
            time6 = time.time()-start_time

            start_time = time.time()
            for i_batch in range(n_batch):
                memory.states_x.append(torch.from_numpy(_es[i_batch]).float().to(torch.device("cpu:0")))
                memory.states_e_r0.append(torch.from_numpy(edge_index_r0[i_batch]).to(torch.device("cpu:0")))
                memory.states_e_r1.append(torch.from_numpy(edge_index_r1[i_batch]).to(torch.device("cpu:0")))
                memory.states_e_n.append(torch.from_numpy(edge_index_n[i_batch]).to(torch.device("cpu:0")))
                memory.states_state_.append(h[i_batch].detach().to(torch.device("cpu:0")))
                memory.states_input_.append(input_[i_batch].detach().to(torch.device("cpu:0")))
                memory.action_node.append(action_node[i_batch].to(torch.device("cpu:0")))
                memory.action_sisr.append(action_sisr[i_batch].to(torch.device("cpu:0")))
                memory.logprobs.append(log_probs[i_batch].to(torch.device("cpu:0")))
            memory.rewards.append(rewards)
            memory.is_terminals.append(i_iter==(n_iter-1))
            time7 = time.time()-start_time
            # print(time1, time2, time3, time4, time5, time6, time7)

        n_iter = min(n_iter+1, n_iter_max)

        ######################################################################
        # 对比结果，训练，清空memory，保存策略
        rl_bests = np.mean([env.best_dist for env in envs[:n_benchs]])
        if cheat:
            print("Mean RL operator", rl_bests, "Evaluation!")
        else:
            print("Mean RL operator", rl_bests)
        agent.update(memory, n_nodes+1)
        memory.clear_memory()
        agent.save(save_path)
        print("Model saved in", save_path)
        print("Training time:", (time.time()-train_start_time)/3600.0, "hrs")
        print("-----------------------------------------")

    os.remove(state_path)
    os.remove(distMat_path)
    shutil.rmtree(data_root)
    _=[os.remove(filename) for filename in os.listdir() if ".class" in filename]
