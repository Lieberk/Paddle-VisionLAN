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
import time
import argparse
from utils import *
from collections import OrderedDict
import sys
import os
import paddle
import paddle.optimizer as optim


class Logger(object):
    def __init__(self, filename="Default.txt"):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


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


def Zero_Grad(model):
    model.clear_gradients()


def Updata_Parameters(optimizers, frozen):
    for i in range(0, len(optimizers)):
        if i not in frozen:
            optimizers[i].step()


def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])
    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    return train_loader, test_loader


def load_network():
    if not cfgs.global_cfgs['cuda']: paddle.device.set_device("cpu")
    paddle.distributed.init_parallel_env()
    model_VL = cfgs.net_cfgs['VisualLAN'](**cfgs.net_cfgs['args'])
    model_VL = paddle.DataParallel(model_VL)
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


def generate_optimizer(model):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=20.0)
    if cfgs.global_cfgs['step'] != 'LF_2':
        scheduler = cfgs.optimizer_cfgs['optimizer_0_scheduler'](**cfgs.optimizer_cfgs['optimizer_0_scheduler_args'])
        out = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, grad_clip=clip)
        return out, scheduler
    else:
        id_mlm = id(model.MLM_VRM.MLM.parameters())
        id_pre_mlm = id(model.MLM_VRM.Prediction.pp_share.parameters()) + id(model.MLM_VRM.Prediction.w_share.parameters())
        id_total = id_mlm + id_pre_mlm
        cfgs.optimizer_cfgs['optimizer_0_scheduler_args']['learning_rate'] *= 0.1
        scheduler = cfgs.optimizer_cfgs['optimizer_0_scheduler'](**cfgs.optimizer_cfgs['optimizer_0_scheduler_args'])
        out = optim.Adam(parameters=filter(lambda p: id(p) != id_total, model.parameters()), learning_rate=scheduler, grad_clip=clip)
        return out, scheduler


def _flatten(sources, lengths):
    return paddle.concat([t[:l] for t, l in zip(sources, lengths)])


def _test(test_loader, model, tools, best_acc):
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
    parser.add_argument('--cfg_type', default='LF_1', choices=['LF_1', 'LF_2', 'LA'], help='train step')
    parser.add_argument('--use_gpu', default=True, help='use cuda for training')
    parser.add_argument('--epochs', default=8, type=int, help='number of epochs to train the model')
    parser.add_argument('--output_dir', default='./output/LF_1/', help='output dir')
    parser.add_argument('--batch_size', default=384, type=int, help='batch size')
    parser.add_argument('--pretrained', default=None, help='use pretrained values')
    parser.add_argument('--data_path', default='./datasets/', help='input folder of the models')
    parser.add_argument('--test_batch_size', default=64, type=int, help='test batch size')
    parser.add_argument('--show_interval', default=200, type=int, help='show interval')
    parser.add_argument('--test_interval', default=2000, type=int, help='test interval')
    args = parser.parse_args()
    if args.cfg_type == 'LF_1':
        import cfgs.cfgs_LF_1 as cfgs
    elif args.cfg_type == 'LF_2':
        import cfgs.cfgs_LF_2 as cfgs
    elif args.cfg_type == 'LA':
        import cfgs.cfgs_LA as cfgs
    cfgs.global_cfgs['show_interval'] = args.show_interval
    cfgs.global_cfgs['test_interval'] = args.test_interval
    cfgs.global_cfgs['cuda'] = args.use_gpu
    cfgs.global_cfgs['epoch'] = args.epochs
    cfgs.dataset_cfgs['dataset_train_args']['roots'] = \
        [os.path.join(args.data_path, 'train', 'SynText'), os.path.join(args.data_path, 'train', 'MJSynth')]
    cfgs.dataset_cfgs['dataloader_train']['batch_size'] = args.batch_size
    cfgs.net_cfgs['init_state_dict'] = args.pretrained
    cfgs.saving_cfgs['saving_path'] = args.output_dir
    cfgs.dataset_cfgs['dataset_test_args']['roots'] = [os.path.join(args.data_path, 'evaluation', 'Sumof6benchmarks')]
    cfgs.dataset_cfgs['dataloader_test']['batch_size'] = args.test_batch_size

    model = load_network()
    optimizer, optimizer_scheduler = generate_optimizer(model)
    criterion_CE = paddle.nn.CrossEntropyLoss()
    L1_loss = paddle.nn.L1Loss()
    train_loader, test_loader = load_dataset()
    # tools prepare
    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                             cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_rem = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_sub = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    # train
    total_iters = len(train_loader)
    loss_show = 0
    time_cal = 0
    ratio_res = 0.5
    ratio_sub = 0.5
    best_acc = 0
    loss_ori_show = 0
    loss_mas_show = 0

    if not os.path.exists(f"logs/{cfgs.global_cfgs['step']}"):
        os.makedirs(f"logs/{cfgs.global_cfgs['step']}")
    logger = Logger(f"logs/{cfgs.global_cfgs['step']}/train.log")

    if not os.path.isdir(cfgs.saving_cfgs['saving_path']):
        os.mkdir(cfgs.saving_cfgs['saving_path'])
    for nEpoch in range(0, cfgs.global_cfgs['epoch']):
        # training log
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        batch_past = 0

        for batch_idx, sample_batched in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            # data_prepare
            data = sample_batched['image']
            label = sample_batched['label']  # original string
            label_res = sample_batched['label_res']  # remaining string
            label_sub = sample_batched['label_sub']  # occluded character
            label_id = sample_batched['label_id']  # character index
            target = encdec.encode(label)
            label_flatten, length = flatten_label(target)
            # prediction
            text_pre, text_rem, text_mas, att_mask_sub = model(data, label_id, cfgs.global_cfgs['step'])
            # loss_calculation
            if cfgs.global_cfgs['step'] == 'LF_1':
                text_pre = _flatten(text_pre, length)
                pre_ori, label_ori = train_acc_counter.add_iter(text_pre, length.cast('int64'), length, label) 
                loss_ori = criterion_CE(text_pre, label_flatten)
                loss = loss_ori
            else:
                target_res = encdec.encode(label_res)
                target_sub = encdec.encode(label_sub)
                label_flatten_res, length_res = flatten_label(target_res)
                label_flatten_sub, length_sub = flatten_label(target_sub)
                text_pre = _flatten(text_pre, length)
                text_rem = _flatten(text_rem, length_res)
                text_mas = _flatten(text_mas, length_sub)
                pre_ori, label_ori = train_acc_counter.add_iter(text_pre, length.cast('int64'), length, label)
                pre_rem, label_rem = train_acc_counter_rem.add_iter(text_rem, length_res.cast('int64'), length_res, label_res)
                pre_sub, label_sub = train_acc_counter_sub.add_iter(text_mas, length_sub.cast('int64'), length_sub, label_sub)

                loss_ori = criterion_CE(text_pre, label_flatten)
                loss_res = criterion_CE(text_rem, label_flatten_res)
                loss_mas = criterion_CE(text_mas, label_flatten_sub)
                loss = loss_ori + loss_res * ratio_res + loss_mas * ratio_sub
                loss_ori_show += loss_res
                loss_mas_show += loss_mas
            # loss for display
            loss_show += loss
            # optimize
            Zero_Grad(model)
            loss.backward()
            optimizer.step()

            train_run_cost += time.time() - train_start
            total_samples += data.shape[0]
            batch_past += 1
            # display
            if batch_idx % cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
                loss_show = loss_show / cfgs.global_cfgs['show_interval']
                print(
                    'Epoch: {}, Iter: {}/{}, Loss VisionLAN: {:0.4f}, avg_reader_cost: {:.4f}, avg_batch_cost: {:.4f}, avg_ips: {:.4f}'.format(
                        nEpoch,
                        batch_idx,
                        total_iters,
                        loss_show.item(),
                        train_reader_cost / batch_past,
                        (train_reader_cost + train_run_cost) / batch_past,
                        total_samples / (train_reader_cost + train_run_cost)
                    ))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
                batch_past = 0
                loss_show = 0
                train_acc_counter.show()
                if cfgs.global_cfgs['step'] != 'LF_1':
                    print(
                        'orignial: {}, mask_character pre/gt: {}/{}, other pre/gts: {}/{}'.format(
                            label[0],
                            pre_sub[0],
                            label_sub[0],
                            pre_rem[0],
                            label_rem[0]))
                    loss_mas_show = loss_mas_show / cfgs.global_cfgs['show_interval']
                    loss_ori_show = loss_ori_show / cfgs.global_cfgs['show_interval']
                    print('loss for mas/rem: {}/{}'.format(loss_mas_show.item(), loss_ori_show.item()))
                    loss_ori_show = 0
                    loss_mas_show = 0
            sys.stdout.flush()
            # evaluation during training
            if batch_idx % cfgs.global_cfgs['test_interval'] == 0 and batch_idx != 0:
                print('Testing during training:')
                best_acc, if_save = _test((test_loader),
                                          model,
                                          [encdec,
                                           flatten_label,
                                           test_acc_counter], best_acc)
                if if_save:
                    paddle.save(model.state_dict(),
                               cfgs.saving_cfgs['saving_path'] + 'best_acc_M.pdparams')
            reader_start = time.time()
        # save each epoch
        if nEpoch % cfgs.saving_cfgs['saving_epoch_interval'] == 0:
            paddle.save(model.state_dict(),
                       cfgs.saving_cfgs['saving_path'] + 'E{}.pdparams'.format(
                           nEpoch))
        optimizer_scheduler.step()
    logger.reset()
