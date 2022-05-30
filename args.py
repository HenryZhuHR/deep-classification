

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # hardware
    parser.add_argument('--device', type=str, default='cuda:0,1',
                        help='cuda:0,1,2')
    # dataset
    
    # train
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    return parser.parse_args()
