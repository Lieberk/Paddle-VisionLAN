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
import paddle.nn as nn
import paddle
import paddle.nn.functional as F
import numpy as np


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class PositionalEncoding(nn.Layer):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return paddle.to_tensor(sinusoid_table, dtype='float32').unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.shape[1]].clone()


class ScaledDotProductAttention(nn.Layer):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(axis=2)

    def forward(self, q, k, v, mask=None):
        attn = paddle.bmm(q, k.transpose([0, 2, 1]))
        attn = attn / self.temperature
        if mask is not None:
            attn = masked_fill(attn, mask, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = paddle.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Layer):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=np.sqrt(2.0 / (d_model + d_k))))
        self.w_ks = nn.Linear(d_model, n_head * d_k, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=np.sqrt(2.0 / (d_model + d_k))))
        self.w_vs = nn.Linear(d_model, n_head * d_v, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=np.sqrt(2.0 / (d_model + d_v))))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model, weight_attr=paddle.nn.initializer.XavierNormal(fan_in=None))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        residual = q
        q = self.w_qs(q).reshape([sz_b, len_q, n_head, d_k])  # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k).reshape([sz_b, len_k, n_head, d_k])
        v = self.w_vs(v).reshape([sz_b, len_v, n_head, d_v])
        q = q.transpose([2, 0, 1, 3]).reshape([-1, len_q, d_k])  # (n*b) x lq x dk
        k = k.transpose([2, 0, 1, 3]).reshape([-1, len_k, d_k])  # (n*b) x lk x dk
        v = v.transpose([2, 0, 1, 3]).reshape([-1, len_v, d_v])  # (n*b) x lv x dv
        mask = mask.repeat_interleave(n_head, 1, 1) if mask is not None else None  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.reshape([n_head, sz_b, len_q, d_v])
        output = output.transpose([1, 2, 0, 3]).reshape([sz_b, len_q, n_head * d_v])  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Layer):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1D(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1D(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose([0, 2, 1])
        x = self.w_2(F.relu(self.w_1(x)))
        x = x.transpose([0, 2, 1])
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class EncoderLayer(nn.Layer):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Transforme_Encoder(nn.Layer):
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, n_position=256):
        super(Transforme_Encoder, self).__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.LayerList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

    def forward(self, enc_output, src_mask, return_attns=False):
        enc_output = self.dropout(self.position_enc(enc_output))  # position embeding
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        enc_output = self.layer_norm(enc_output)
        return enc_output,


class Transforme_Encoder_light(nn.Layer):
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=768, dropout=0.1, n_position=256):
        super(Transforme_Encoder_light, self).__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.LayerList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

    def forward(self, enc_output, src_mask, return_attns=False):
        enc_output = self.dropout(self.position_enc(enc_output))  # position embeding
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        enc_output = self.layer_norm(enc_output)
        return enc_output,


class PP_layer(nn.Layer):
    def __init__(self, n_dim=512, N_max_character=25, n_position=256):
        super(PP_layer, self).__init__()
        self.character_len = N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, N_max_character)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(axis=2)

    def forward(self, enc_output):
        reading_order = paddle.arange(self.character_len, dtype='int64')
        reading_order = reading_order.unsqueeze(0).expand([enc_output.shape[0], -1])  # (S,) -> (B, S)
        reading_order = self.f0_embedding(reading_order)  # b,25,512
        # calculate attention
        t = self.w0(reading_order.transpose([0, 2, 1]))  # b,512,256
        t = self.active(t.transpose([0, 2, 1]) + self.wv(enc_output))  # b,256,512
        t = self.we(t)  # b,256,25
        t = self.softmax(t.transpose([0, 2, 1]))  # b,25,256
        g_output = paddle.bmm(t, enc_output)  # b,25,512
        return g_output, t


class Prediction(nn.Layer):
    def __init__(self, n_dim=512, n_class=37, N_max_character=25, n_position=256, GSRM_layer=4, nchannel=512):
        super(Prediction, self).__init__()
        self.pp = PP_layer(N_max_character=N_max_character, n_position=n_position)
        self.pp_share = PP_layer(N_max_character=N_max_character, n_position=n_position)
        self.w_vrm = nn.Linear(n_dim, n_class)  # output layer
        self.w_share = nn.Linear(n_dim, n_class)  # output layer
        self.nclass = n_class

    def forward(self, cnn_feature, f_res, f_sub, Train_is=False, use_mlm=True):
        if Train_is:
            if not use_mlm:
                g_output, attn = self.pp(cnn_feature)  # b,25,512
                g_output = self.w_vrm(g_output)
                f_res = 0
                f_sub = 0
                return g_output, f_res, f_sub
            g_output, attn = self.pp(cnn_feature)  # b,25,512
            f_res, _ = self.pp_share(f_res)
            f_sub, _ = self.pp_share(f_sub)
            g_output = self.w_vrm(g_output)
            f_res = self.w_share(f_res)
            f_sub = self.w_share(f_sub)
            return g_output, f_res, f_sub
        else:
            g_output, attn = self.pp(cnn_feature)  # b,25,512
            g_output = self.w_vrm(g_output)
            return g_output
