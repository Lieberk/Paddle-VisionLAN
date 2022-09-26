from __future__ import print_function
import paddle.vision.transforms as transforms
from utils import *
import cfgs.cfgs_eval as cfgs
from collections import OrderedDict
from PIL import Image
import argparse


def Train_or_Eval(model, state='Train'):
    if state == 'Train':
        model.train()
    else:
        model.eval()


def load_network():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training')
    parser.add_argument('--pretrained', default='./output/LA/best_acc_M.pdparams', help='use pretrained values')
    parser.add_argument('--img_file', default='./images/demo1.png', help='input folder of the models')
    args = parser.parse_args()
    cfgs.net_cfgs['init_state_dict'] = args.pretrained

    model = load_network()
    img_width = 256
    img_height = 64
    transf = transforms.ToTensor()
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    Train_or_Eval(model, 'Eval')
    img = Image.open(args.img_file).convert('RGB')
    img = img.resize((img_width, img_height))
    img = transf(img)
    img = paddle.unsqueeze(img, axis=0)
    target = ''
    output, out_length = model(img, target, '', False)
    pre_string = test_acc_counter.convert(output, out_length)
    print('pre_string:', pre_string[0])
