# coding:utf-8
import paddle.optimizer as optim
from data.dataset_scene import *
import paddle.vision.transforms as transforms
from VisionLAN import *

global_cfgs = {
    'state': 'Train',
    'epoch': 4,
    'show_interval': 200,
    'test_interval': 2000,
    'step': 'LF_2',
    'cuda': True,
}
dataset_cfgs = {
    'dataset_train': lmdbDataset,
    'dataset_train_args': {
        'roots': [
            './datasets/train/SynText',
            './datasets/train/MJSynth',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Train',
    },
    'dataloader_train': {
        'batch_size': 220,
        'shuffle': True,
        'num_workers': 0,
        'use_shared_memory': True,
    },

    'dataset_test': lmdbDataset,
    'dataset_test_args': {
        'roots': [
            './datasets/evaluation/Sumof6benchmarks'
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataloader_test': {
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 0,
        'use_shared_memory': True,
    },
    'case_sensitive': False,
    'dict_dir': './dict/dic_36.txt'
}

net_cfgs = {
    'VisualLAN': VisionLAN,
    'args': {
        'strides': [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (1, 1)],
        'input_shape': [3, 64, 256],  # C x H x W
    },

    'init_state_dict': './output/LF_1/best_acc_M.pdparams',
}


optimizer_cfgs = {
    'optimizer_0': optim.Adam,
    'optimizer_0_scheduler': optim.lr.MultiStepDecay,
    'optimizer_0_scheduler_args': {
        'learning_rate': 0.0001,
        'milestones': [2],
        'gamma': 0.1,
    },
}

saving_cfgs = {
    'saving_epoch_interval': 1,
    'saving_path': './output/LF_2/',

}
