import os
import argparse
import yaml
from main_util import *
from train import *

from PIL import Image

parser = argparse.ArgumentParser(description='Train the gazeclip network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./logs', dest='dir_log')
parser.add_argument('--scope', default='gazeclip', dest='scope')
parser.add_argument('--name_data', type=str, default='DyGaze', dest='name_data')
parser.add_argument('--dir_data', default='file path to /D_StaticGazes', dest='dir_data')

parser.add_argument('--num_epoch', type=int,  default=40, dest='num_epoch')

parser.add_argument('--batch_size', type=int, default=40, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=0.000008, dest='lr_G')

parser.add_argument('--beta1', default=0.9, dest='beta1')

parser.add_argument('--num_freq_disp', type=int,  default=500, dest='num_freq_disp')

parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')

PARSER = Parser(parser)

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

if __name__ == '__main__':

    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()


    TRAINER = Train(ARGS)
    if ARGS.mode == 'train' or ARGS.mode == 'pre':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()








