# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding:utf-8
from __future__ import print_function
from paddle.io import DataLoader
from utils import *
import cfgs.cfgs_eval as cfgs
from collections import OrderedDict
import argparse
import os


def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.shape[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)
    label_flatten = paddle.to_tensor(label_flatten, dtype='int64')
    label_length = paddle.to_tensor(label_length, dtype='int64')
    return (label_flatten, label_length)


def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])

    test_data_all = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_all'])
    test_loader_all = DataLoader(test_data_all, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setIC13 = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsIC13'])
    test_loaderIC13 = DataLoader(test_data_setIC13, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setIC15 = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsIC15'])
    test_loaderIC15 = DataLoader(test_data_setIC15, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setSVT = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsSVT'])
    test_loaderSVT = DataLoader(test_data_setSVT, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setSVTP = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsSVTP'])
    test_loaderSVTP = DataLoader(test_data_setSVTP, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setCUTE = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsCUTE'])
    test_loaderCUTE = DataLoader(test_data_setCUTE, **cfgs.dataset_cfgs['dataloader_test'])

    # pdb.set_trace()
    return (train_loader, test_loader_all, test_loader, test_loaderIC13, test_loaderIC15, test_loaderSVT, test_loaderSVTP, test_loaderCUTE)


def load_network():
    if not cfgs.global_cfgs['cuda']: paddle.device.set_device("cpu")
    model_VL = cfgs.net_cfgs['VisualLAN'](**cfgs.net_cfgs['args'])
    if cfgs.net_cfgs['init_state_dict'] is not None:
        fe_state_dict_ori = paddle.load(cfgs.net_cfgs['init_state_dict'])
        fe_state_dict = OrderedDict()
        for k, v in fe_state_dict_ori.items():
            fe_state_dict[k] = v
        model_dict_fe = model_VL.state_dict()
        state_dict_fe = {k: v for k, v in fe_state_dict.items() if k in model_dict_fe.keys()}
        model_dict_fe.update(state_dict_fe)
        model_VL.load_dict(model_dict_fe)
    return model_VL


def _test(test_loader, model, tools, best_acc, string_name):
    print('------' + string_name + '--------')
    for sample_batched in test_loader:
        data = sample_batched['image']
        label = sample_batched['label']
        target = tools[0].encode(label)
        label_flatten, length = tools[1](target)
        output, out_length = model(data, target, '', False)
        tools[2].add_iter(output, out_length, length, label)
    best_acc, change = tools[2].show_test(best_acc)
    return best_acc, change


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training')
    parser.add_argument('--use_gpu', default=True, help='use cuda for training')
    parser.add_argument('--pretrained', default='./output/LA/best_acc_M.pdparams', help='use pretrained values')
    parser.add_argument('--data_path', default='./datasets/', help='input folder of the models')
    args = parser.parse_args()
    cfgs.global_cfgs['cuda'] = args.use_gpu
    cfgs.net_cfgs['init_state_dict'] = args.pretrained
    cfgs.dataset_cfgs['dataset_train_args']['roots'] = \
        [os.path.join(args.data_path, 'train', 'SynText'), os.path.join(args.data_path, 'train', 'MJSynth')]
    cfgs.dataset_cfgs['dataset_test_all']['roots'] = [os.path.join(args.data_path, 'evaluation', 'Sumof6benchmarks')]
    cfgs.dataset_cfgs['dataset_test_args']['roots'] = [os.path.join(args.data_path, 'evaluation', 'IIIT5K')]
    cfgs.dataset_cfgs['dataset_test_argsIC13']['roots'] = [os.path.join(args.data_path, 'evaluation', 'IC13')]
    cfgs.dataset_cfgs['dataset_test_argsIC15']['roots'] = [os.path.join(args.data_path, 'evaluation', 'IC15')]
    cfgs.dataset_cfgs['dataset_test_argsSVT']['roots'] = [os.path.join(args.data_path, 'evaluation', 'SVT')]
    cfgs.dataset_cfgs['dataset_test_argsSVTP']['roots'] = [os.path.join(args.data_path, 'evaluation', 'SVTP')]
    cfgs.dataset_cfgs['dataset_test_argsCUTE']['roots'] = [os.path.join(args.data_path, 'evaluation', 'CUTE')]

    model = load_network()
    train_loader, test_loader_all, test_loader, test_loaderIC13, test_loaderIC15, test_loaderSVT, test_loaderSVTP, test_loaderCUTE = load_dataset()
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])

    if cfgs.global_cfgs['state'] == 'Test':
        _test((test_loader_all),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='Average on 6 benchmarks')

        _test((test_loader),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='IIIT')
        _test((test_loaderIC13),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='IC13')
        _test((test_loaderIC15),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='IC15')
        _test((test_loaderSVT),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='SVT')
        _test((test_loaderSVTP),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='SVTP')
        _test((test_loaderCUTE),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='CUTE')
        exit()
