import paddle
import paddle.nn.functional as F
import editdistance as ed
from paddle.vision.transforms import BaseTransform
from PIL import Image
import numpy as np


class cha_encdec():
    def __init__(self, dict_file, case_sensitive=True):
        self.dict = []
        self.case_sensitive = case_sensitive
        lines = open(dict_file, 'r').readlines()
        for line in lines:
            self.dict.append(line.replace('\n', ''))

    def encode(self, label_batch):
        max_len = max([len(s) for s in label_batch])
        out = np.zeros([len(label_batch), max_len + 1], dtype='int64')
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = np.array([self.dict.index(char.lower()) if char.lower() in self.dict else len(self.dict)
                                                for char in label_batch[i]]) + 1
            else:
                cur_encoded = np.array([self.dict.index(char) if char in self.dict else len(self.dict)
                                                for char in label_batch[i]]) + 1
            out[i][:len(cur_encoded)] = cur_encoded
        return paddle.to_tensor(out)

    def decode(self, net_out, length):
        out = []
        out_prob = []
        net_out = F.softmax(net_out, axis=1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:, 0].tolist()
            current_text = ''.join([self.dict[_ - 1] if _ > 0 and _ <= len(self.dict) else '' for _ in current_idx_list])
            current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
            current_probability = paddle.exp(paddle.log(current_probability).sum() / current_probability.shape[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return (out, out_prob)


class Attention_AR_counter():
    def __init__(self, display_string, dict_file, case_sensitive):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.display_string = display_string
        self.case_sensitive = case_sensitive
        self.de = cha_encdec(dict_file, case_sensitive)

    def clear(self):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.

    def add_iter(self, output, out_length, label_length, labels):
        self.total_samples += label_length.shape[0]
        prdt_texts, prdt_prob = self.de.decode(output, out_length)
        for i in range(0, len(prdt_texts)):
            if not self.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            self.distance_C += ed.eval(labels[i], prdt_texts[i])
            self.distance_W += ed.eval(l_words, p_words)
            self.total_C += len(labels[i])
            self.total_W += len(l_words)
            self.correct = self.correct + 1 if labels[i] == prdt_texts[i] else self.correct
        return prdt_texts, labels

    def show(self):
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W))
        self.clear()

    def show_test(self, best_acc, change=False):
        print(self.display_string)
        if self.total_samples == 0:
            pass
        if (self.correct / self.total_samples) >= best_acc:
            best_acc = np.copy(self.correct / self.total_samples)
            change = True
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}, best_acc: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W, best_acc))

        self.clear()
        return best_acc, change

    def convert(self, output, out_length):
        prdt_texts, prdt_prob = self.de.decode(output, out_length)
        return prdt_texts


class ToPILImage(BaseTransform):
    def __init__(self, keys=None):
        super().__init__(keys)

    def _apply_image(self, pic, mode=None):
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

            # check number of channels
            if pic.shape[-3] > 4:
                raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-3]))

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

            # check number of channels
            if pic.shape[-1] > 4:
                raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-1]))

        npimg = pic
        if isinstance(pic, paddle.Tensor):
            if "float" in str(pic.numpy().dtype) and mode != 'F':
                pic = pic.multiply(paddle.to_tensor(255.0)).cast('uint8')
            npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                            'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if mode is not None and mode != expected_mode:
                raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                                 .format(mode, np.dtype, expected_mode))
            mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if mode is not None and mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if mode is not None and mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if mode is not None and mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGB'

        if mode is None:
            raise TypeError('Input type {} is not supported'.format(npimg.dtype))

        return Image.fromarray(npimg, mode=mode)
