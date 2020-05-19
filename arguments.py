import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_iter', default=150)
    parser.add_argument('-b', '--n_batch', default=128)
    parser.add_argument('--benchs', default=16)
    parser.add_argument('--epoch', default=20000)
    parser.add_argument('--anchors', default=2)
    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--k_epoch', default=2)
    parser.add_argument('--eps_clip', default=0.2)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--node_dim', default=128)
    parser.add_argument('--critic_dims', default=256)
    parser.add_argument('--init_T', default=100.0)
    parser.add_argument('--final_T', default=1.0)
    parser.add_argument('--device', default="cpu")
    return parser.parse_args()
