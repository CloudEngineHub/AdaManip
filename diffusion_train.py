import argparse
from socket import IP_DEFAULT_MULTICAST_LOOP
import ipdb

def get_args():
    # use parser to get args
    parser = argparse.ArgumentParser()

    parser.add_argument('--action_dim', type=int, default=9)

    parser.add_argument('--dataset_path', type=str, nargs='+')
    parser.add_argument('--pred_horizon', type=int, default=4)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=1)
    parser.add_argument('--dof_dim', type=int, default=0)
    parser.add_argument('--num_diffusion_iters', type=int, default=100)
    parser.add_argument('--DDIM', action='store_true', default=False)
    parser.add_argument('--discrete', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--load_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_rate', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--logdir', type=str, default='logs')
    # seg pointnet para
    parser.add_argument('--input_feat', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default='128')
    # transformer para
    # parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_cond_layers', type=int, default=0) # >0: use transformer encoder for cond, otherwise use MLP

    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_emb', type=int, default=256)
    parser.add_argument('--p_drop_emb', type=float, default=0.0)
    parser.add_argument('--p_drop_attn', type=float, default=0.3)
    
    parser.add_argument('--causal_attn', action='store_true', default=True)
    parser.add_argument('--time_as_cond', action='store_true', default=True)  # if false, use BERT like encoder only arch, time as input
    parser.add_argument('--pred_action_steps_only', action='store_true', default=False)

    # optimizer
    parser.add_argument('--transformer_weight_decay', type=float, default=1.0e-3)
    parser.add_argument('--obs_encoder_weight_decay', type=float, default=1.0e-6)
    parser.add_argument('--learning_rate', type=float, default=1.0e-4)

    # control
    parser.add_argument('--resume', action='store_true', default=False)
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    from diffusion_policy.diffusion_policy_new import DiffusionPolicy
    policy = DiffusionPolicy(args)
    # from diffusion_policy.diffusion_policy_transformer import DiffusionPolicyTran
    # policy = DiffusionPolicyTran(args)
    
    policy.train()