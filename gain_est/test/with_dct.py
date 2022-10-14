import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
import soundfile as sf
import einops
import math
from losses import si_snr
from functional import mdct, imdct, stdct, istdct
from layers import ConvSTFT,ConviSTFT, ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm , CausalConvTranspose2d, CausalConv2d


class Conv2Dblock(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 kernel_size,
                 stride,
                 padding=(0,0)):
        super(Conv2Dblock, self).__init__()

        #self.conv_2d = ComplexConv2d(input_ch,output_ch,kernel_size=kernel_size,stride=stride,padding=padding)
        self.conv_2d = CausalConv2d(input_ch,output_ch,kernel_size=kernel_size,stride=stride,padding=padding)
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation = nn.PReLU()
        # self.activation = nn.SiLU()
        #self.alpha_param = nn.parameter.Parameter(torch.ones(1,output_ch,1,1))
    
    def snake(self, x, a):
        acts = x + (1 / a) * (torch.sin(a * x) ** 2)  # Snake1D
        return acts
    
    def forward(self,inputs):
        output = self.conv_2d(inputs)
        output = self.normalization(output)
        # output = self.snake(output,self.alpha_param)
        output = self.activation(output)
        return output
    
    
    
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

class self_attention(nn.Module):
    def __init__(self, c=64, att='time', causal=True):
        super(self_attention, self).__init__()
        self.d_c = c // 2
        self.att = att
        att_list = ['time','freq']
        if self.att=='time':
            self.t_qkv1 = Conv2Dblock(c, self.d_c, kernel_size=(1, 1), stride=(1, 1))
        elif self.att=='freq':
            self.f_qkv1 = Conv2Dblock(c, self.d_c, kernel_size=(1,1),stride=(1,1))
        else:
            assert self.att in att_list
        self.proj = Conv2Dblock(self.d_c,c,kernel_size=(1,1),stride=(1,1))
        self.causal = causal
        
    def forward(self,inp):
        score = 0
        if self.att =='time':
            qkv = self.t_qkv1(inp)
            score = torch.einsum('bcft,bcfy->bfty', qkv, qkv) / (self.d_c ** 0.5)
            score = torch.einsum('bfty,bcfy->bcft', [score, qkv])
        elif self.att=='freq':
            qkv = self.f_qkv1(inp)
            score = torch.einsum("bcft,bcyt->btfy", qkv, qkv) / (self.d_c ** 0.5)
            score = torch.einsum('btfy,bcyt->bcft', [score, qkv])
        else:
            return 0
        mask_value = max_neg_value(score)
        if self.causal:
            i, j = score.shape[-2:]
            mask = torch.ones(i, j, device=score.device).triu_(j - i + 1).bool()
            score.masked_fill_(mask, mask_value)
        score = score.softmax(dim=-1)
        score = self.proj(score)
        out = score+inp
        return out

class ASA(nn.Module):
    def __init__(self, c=64, causal=True):
        super(ASA, self).__init__()
        self.d_c = c//4
        self.f_qkv = Conv2Dblock(c, self.d_c*3,kernel_size=(1,1),stride=(1,1))
        self.t_qk = Conv2Dblock(c, self.d_c*2, kernel_size=(1, 1),stride=(1,1))
        self.proj = Conv2Dblock(self.d_c, c,kernel_size=(1,1),stride=(1,1))
        self.causal = causal

    def forward(self, inp):

        """
        inp: B C F T
        """
        # f-attention
        f_qkv = self.f_qkv(inp)
        qf, kf, v = tuple(einops.rearrange(
            f_qkv, "b (c k) f t->k b c f t", k=3))
        f_score = torch.einsum("bcft,bcyt->btfy", qf, kf) / (self.d_c**0.5)
        f_score = f_score.softmax(dim=-1)
        f_out = torch.einsum('btfy,bcyt->bcft', [f_score, v])
        # t-attention
        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
        t_score = torch.einsum('bcft,bcfy->bfty', [qt, kt]) / (self.d_c**0.5)
        mask_value = max_neg_value(t_score)
        if self.causal:
            i, j = t_score.shape[-2:]
            mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
            t_score.masked_fill_(mask, mask_value)
        t_score = t_score.softmax(dim=-1)
        t_out = torch.einsum('bfty,bcfy->bcft', [t_score, f_out])
        out = self.proj(t_out)
        return out + inp

class Encoder_farend(nn.Module):
    def __init__(self):
        super(Encoder_farend, self).__init__()
        self.conv1 = Conv2Dblock(4, 64, kernel_size=(5, 3),stride=(4, 1))
    def forward(self,inputs):
        output1 = self.conv1(inputs)
        return (inputs,output1)

class Encoder_nearend(nn.Module):
    def __init__(self):
        super(Encoder_nearend, self).__init__()
        self.conv1 = Conv2Dblock(2, 16, kernel_size=(5, 3), stride=(1, 1))
        self.conv2 = Conv2Dblock(16, 32, kernel_size=(5, 3), stride=(2, 1))
        self.conv3 = Conv2Dblock(32, 64, kernel_size=(5, 3), stride=(2, 1))
        # self.conv4 = Conv2Dblock(32*2, 64*2, kernel_size=(5, 3), stride=(2, 1), padding=(2, 2))
        # self.conv1 = Conv2Dblock(4, 16, kernel_size=(3, 5),stride=(1, 1),padding=(1,2))
        # self.conv2 = Conv2Dblock(16, 32, kernel_size=(3, 5),stride=(2,1),padding=(1,2))
        # self.conv3 = Conv2Dblock(32, 64, kernel_size=(3, 5), stride=(2,1),padding=(1,2))
    def forward(self,inputs):
        output1 = self.conv1(inputs)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        # output4 = self.conv4(output3)
        return (inputs,output1,output2,output3)



class Decoder_nearend(nn.Module):
    def __init__(self):
        super(Decoder_nearend,self).__init__()
        self.deconv1 = CausalConvTranspose2d(64, 32, kernel_size=(5, 3), stride=(2, 1))
        self.deconv2 = CausalConvTranspose2d(32,16,kernel_size=(5,3),stride=(2,1))
        self.deconv3 = CausalConvTranspose2d(16,2,kernel_size=(5,3),stride=(1,1))
        self.conv1_1 = Conv2Dblock(64, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv1_2 = Conv2Dblock(64, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_1 = Conv2Dblock(32,16, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_2 = Conv2Dblock(32,16, kernel_size=(1, 1), stride=(1, 1))
        self.conv3_1 = Conv2Dblock(4,2,kernel_size=(1,1),stride=(1,1))
        self.conv3_2 = Conv2Dblock(4,2, kernel_size=(1, 1), stride=(1, 1))

        self.conv_4 = Conv2Dblock(2,1,kernel_size=(1,1),stride=(1,1))

    def forward(self,inputs,encoder_input):
        output_11 = self.deconv1(inputs)
        output_12 = torch.cat([output_11,encoder_input[2]],axis=1)
        output_13 = self.conv1_1(output_12)
        output_14 = output_13*encoder_input[2]
        output_15 = torch.cat([output_14,output_11],axis=1)
        output_16 = self.conv1_2(output_15)
        output_1 = output_16 + output_11

        output_21 = self.deconv2(output_1)
        output_22 = torch.cat([output_21, encoder_input[1]], axis=1)
        output_23 = self.conv2_1(output_22)
        output_24 = output_23 * encoder_input[1]
        output_25 = torch.cat([output_24, output_21],axis=1)
        output_26 = self.conv2_2(output_25)
        output_2 = output_26 + output_21

        output_31 = self.deconv3(output_2)
        output_32 = torch.cat([output_31, encoder_input[0]], axis=1)
        output_33 = self.conv3_1(output_32)
        output_34 = output_33 * encoder_input[0]
        output_35 = torch.cat([output_34, output_31],axis=1)
        output_36 = self.conv3_2(output_35)
        output_3 = output_36 + output_31

        output = self.conv_4(output_3)
        return output

class RA_Block(nn.Module):
    def __init__(self,
                 attention=True):
        super(RA_Block,self).__init__()

        self.res_block1 = CausalConv2d(64,64,kernel_size=(7,5),stride=(1,1))
        self.res_block2 = CausalConv2d(64,64,kernel_size=(7,5),stride=(1,1))
        self.activation = nn.Sigmoid()

        self.attention = attention

        if self.attention:
            # self.freq_conv_block_qkv = ComplexConv2d(64,32,kernel_size=(1,1),stride=(1,1))
            # self.freq_conv_block2 = ComplexConv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
            # # self.freq_attention_block = LocalAttention(dim=self.F,window_size=256,causal=True,autopad=True,dropout=0.1)
            #
            # self.time_conv_block_qkv = ComplexConv2d(64,32,kernel_size=(1,1),stride=(1,1))
            # self.time_conv_block2 = ComplexConv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
            # # self.time_attention_block = LocalAttention(dim=self.T,window_size=256,causal=True,autopad=True,dropout=0.1)
            #
            # self.conv_block = Conv2Dblock(3*64,64,kernel_size=(1,1),stride=(1,1))
            self.attention_block = ASA(c=64)
            self.conv_block = Conv2Dblock(64, 64, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.conv_block = Conv2Dblock(64,64,kernel_size=(1,1),stride=(1,1))

    def forward(self,inputs):
        f_res = self.res_block1(inputs)
        # f_res = self.res_block2(f_res)
        f_res_gate = self.res_block2(inputs)
        f_res_gate = self.activation(f_res_gate)
        f_res = f_res_gate*f_res
        if self.attention:
            # freq_output = self.freq_attention_block(f_res)
            # time_output = self.time_attention_block(f_res)
            # output = complex_cat([freq_output,f_res,time_output],axis=1)
            # output = self.proj(output)
            output = self.attention_block(f_res)
            output = self.conv_block(output)
        else:
            output = self.conv_block(f_res)
        return output

class interaction_block(nn.Module):
    def __init__(self):
        super(interaction_block,self).__init__()
        self.conv_block = Conv2Dblock(128,64,kernel_size=(1,1),stride=(1,1))
    def forward(self,nearend_feature,echo_feature):
        output = torch.cat([nearend_feature,echo_feature],axis=1)
        output = self.conv_block(output)
        output = output*echo_feature
        output = output + nearend_feature
        return output

class echo_gen(nn.Module):
    def __init__(self,
                win_len=320,
                win_inc=160,
                fft_len=320,
                win_type='hann',
                num_blocks=2):
        super(echo_gen, self).__init__()
        self.num_blocks = num_blocks
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        self.enc_kernel_size = [2,16,32,64]
        fix = True
        self.dct = lambda x: stdct(x, self.fft_len, self.win_inc, self.win_len, center=True, win_type=win_type)
        self.idct = lambda x: istdct(x, self.fft_len, self.win_inc, self.win_len, center=True, win_type=win_type)
        
        # self.nearend_enc = Encoder()
        self.nearend_enc = Encoder_nearend()
        self.farend_enc = Encoder_nearend()
        self.tsb = nn.ModuleList()
        self.tsb_farend = nn.ModuleList()
        self.inter_block = nn.ModuleList()
        self.inter_block_far = nn.ModuleList()
        for i in range(self.num_blocks):
            self.tsb.append(RA_Block(attention=True))
            # self.tsb_farend.append(RA_Block(attention=False))
            # self.inter_block_far.append(interaction_block())
            # self.inter_block.append(interaction_block())
        for i in range(self.num_blocks):
            self.tsb.append(RA_Block(attention=True))
            self.tsb_farend.append(RA_Block(attention=False))
            self.inter_block_far.append(interaction_block())
            self.inter_block.append(interaction_block())
        # self.nearend_dec = Decoder()
        self.nearend_dec = Decoder_nearend()
        self.farend_dec = Decoder_nearend()


    def forward(self,farend,nearend):
        nearend_cspecs = self.dct(nearend)
        farend_cspecs = self.dct(farend)
        nearend_cspecs = nearend_cspecs.unsqueeze(1)
        farend_cspecs = farend_cspecs.unsqueeze(1)
        #print(nearend_cspecs.shape, farend_cspecs.shape)
        nearend_cat = torch.cat([nearend_cspecs,farend_cspecs],1)

        # nearend_enc_output = self.nearend_enc(nearend_cspecs)
        nearend_enc_output = self.nearend_enc(nearend_cat)
        farend_enc_output = self.farend_enc(nearend_cat)
        # farend_enc_output = self.farend_enc(farend_cspecs)

        nearend_ra_output = nearend_enc_output[-1]
        farend_ra_output = farend_enc_output[-1]


        for i in range(self.num_blocks):
            nearend_ra_output = self.tsb[i](nearend_ra_output)
            farend_ra_output = self.tsb_farend[i](farend_ra_output)
            inter_nearend_output = nearend_ra_output
            nearend_ra_output = self.inter_block[i](nearend_ra_output, farend_ra_output)
            farend_ra_output = self.inter_block_far[i](farend_ra_output, inter_nearend_output)
        #     if i%2==0:
        #         farend_ra_output = self.tsb_farend[i//2](farend_ra_output)
        #         inter_nearend_output = nearend_ra_output
        #         nearend_ra_output = self.inter_block[i//2](nearend_ra_output,farend_ra_output)
        #         farend_ra_output = self.inter_block_far[i//2](farend_ra_output,inter_nearend_output)
        #     ee_inputs.append(nearend_ra_output)
        #
        # ee_output = []
        # for i in range(self.num_blocks-1):
        #     temp_output = self.ee_output[i](ee_inputs[i],nearend_enc_output)
        #     temp_output = torch.cat([temp_output[:,0],temp_output[:,1]],1)
        #     ee_output.append(self.istft(temp_output))

        nearend_dec_output = self.nearend_dec(nearend_ra_output,nearend_enc_output)
        farend_dec_output = self.farend_dec(farend_ra_output,farend_enc_output)
        
        nearend_dec_output = nearend_dec_output.squeeze(1)
        farend_dec_output = farend_dec_output.squeeze(1)
        
        nearend_speech_est = self.idct(nearend_dec_output)
        echo_est = self.idct(farend_dec_output)
        # ee_output = self.istft(ee_output)

        return nearend_speech_est , echo_est
    # def loss(self, nearend_speech, nearend_est, nearend_mic, echo_est, type="COMPLEX_MSE"):
    #     b, d, t = nearend_speech.size()
    #
    #
    #     echo_loss = -(si_snr(echo_est[:, 0], nearend_mic[:,0]-echo_est[:, 0]))
    #     nearend_loss = -(si_snr(nearend_est[:, 0], nearend_speech[:, 0]))
    #
    #     total_loss = nearend_loss * 0.9 + echo_loss * 0.1
    def loss(self, nearend_speech, nearend_est, echo, echo_est, type="COMPLEX_MSE"):
        b, d, t = nearend_speech.size()
        if type == "COMPLEX_MSE":
            nearend_est_cspecs = self.stft(nearend_est)
            nearend_speech_cspecs = self.stft(nearend_speech)

            echo_cspecs = self.stft(echo)
            echo_est_cspecs = self.stft(echo_est)

            nearend_est_mag_spec = torch.sqrt(
                nearend_est_cspecs[:, :self.feat_dim, :] ** 2 + nearend_est_cspecs[:, self.feat_dim:, :] ** 2)
            nearend_speech_mag_spec = torch.sqrt(
                nearend_speech_cspecs[:, :self.feat_dim, :] ** 2 + nearend_speech_cspecs[:, self.feat_dim:, :] ** 2)

            echo_est_mag_spec = torch.sqrt(
                echo_est_cspecs[:, :self.feat_dim, :] ** 2 + echo_est_cspecs[:, self.feat_dim:, :] ** 2)
            echo_mag_spec = torch.sqrt(echo_cspecs[:, :self.feat_dim, :] ** 2 + echo_cspecs[:, self.feat_dim:, :] ** 2)

            nearend_est_cprs_mag_spec = nearend_est_mag_spec ** 0.3
            nearend_cprs_mag_spec = nearend_speech_mag_spec ** 0.3

            echo_est_cprs_mag_spec = echo_est_mag_spec ** 0.3
            echo_cprs_mag_spec = echo_mag_spec ** 0.3

            nearend_loss = F.mse_loss(nearend_est_cprs_mag_spec, nearend_cprs_mag_spec)
            echo_loss = F.mse_loss(echo_est_cprs_mag_spec, echo_cprs_mag_spec)
            echo_loss += F.mse_loss(echo[:, 0], echo_est[:, 0], reduction="mean") * 320
            nearend_loss += F.mse_loss(nearend_speech[:, 0], nearend_est[:, 0], reduction="mean") * 320
            echo_loss += (si_snr(echo_est[:, 0], echo[:, 0]))
            nearend_loss += (si_snr(nearend_est[:, 0], nearend_speech[:, 0]))
        elif type == "SI_SNR":
            echo_loss = (si_snr(echo_est[:, 0], echo[:, 0]))
            nearend_loss = (si_snr(nearend_est[:, 0], nearend_speech[:, 0]))
        else:
            echo_loss = F.mse_loss(echo[:, 0], echo_est[:, 0], reduction="mean") * d
            nearend_loss = F.mse_loss(nearend_speech[:, 0], nearend_est[:, 0], reduction="mean") * d
        total_loss = nearend_loss * 0.5 + echo_loss * 0.5
        # total_loss = nearend_loss+ echo_loss

        return total_loss, nearend_loss, echo_loss

if __name__=='__main__':
    from thop import profile, clever_format
    model = echo_gen()
    tmp = (torch.randn(1, 1, 64000), torch.randn(1, 1, 64000))
    macs, params = profile(model, inputs=(tmp), verbose=True)
    macs, params = clever_format([macs, params], "%.3f")
    print('macs: ', macs)
