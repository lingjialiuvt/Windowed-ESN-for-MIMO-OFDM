'''
Copyright (c) 2023, Jiarui Xu, Zhou Zhou, and Lingjia Liu, Virginia Tech.
All rights reserved.
'''

import numpy as np
import os
import scipy.io as sio
from utils.modulation import OFDM_Modulation, Modulation
from utils.power_amplifier import Non_PA

class MIMO_OFDM_signal:

    def __init__(self, MIMO_OFDM_data_set_parameters):

        self.nfft = MIMO_OFDM_data_set_parameters['nfft']
        self.N_t = MIMO_OFDM_data_set_parameters['N_t']
        self.N_r = MIMO_OFDM_data_set_parameters['N_r']
        self.cp_length = MIMO_OFDM_data_set_parameters['CP_length']  # CP length
        self.num_pilot_symbols = MIMO_OFDM_data_set_parameters['training_samples']
        self.num_data_symbols = MIMO_OFDM_data_set_parameters['testing_samples']
        self.total_num_symbols = self.num_pilot_symbols + self.num_data_symbols

        self.EbNo = MIMO_OFDM_data_set_parameters['EbNo']
        self.QAM_order = MIMO_OFDM_data_set_parameters['QAM_order']
        self.gray_code = MIMO_OFDM_data_set_parameters['gray_code']
        self.digit_mod = Modulation(self.QAM_order, self.gray_code)
        self.constellation = self.digit_mod.constellation
        self.ofdm_mod = OFDM_Modulation(self.nfft, self.cp_length)
        self.USE_PA = MIMO_OFDM_data_set_parameters['USE_PA']
        if self.USE_PA:
            self.PA = self.initialize_PA()
        self.noise_var = self.cal_noise_var()

        channel_path = MIMO_OFDM_data_set_parameters['channel_path']
        if 'mat' in os.path.basename(channel_path):
            mat_dict = sio.loadmat(MIMO_OFDM_data_set_parameters['channel_path'])
            channel_all = mat_dict['H']  # (num_channels, Nt, Nr, num_subframes, num_symbols, len_channel)

            self.channel_all = channel_all[:, :, :, :, :self.total_num_symbols, :]
            self.channel_mat_file = True
        else:
            channel_all = np.load(channel_path)  # (Num_channels, N_t, N_r, N_OFDM_Sym, len_channel)
            self.channel_all = channel_all[:, :, :, :self.total_num_symbols, :self.cp_length]
            self.channel_mat_file = False
        if self.channel_all[0].shape[-1] > self.cp_length + 1:
            raise Exception("channel length should be smaller than CP length")

    def generate_data(self, channel_idx=0, subframe_idx=0):
        if self.channel_mat_file:
            self.channel_all[channel_idx, :, :, subframe_idx, ...] = \
                self.normalize_channel(self.channel_all[channel_idx, :, :, subframe_idx, ...])
        else:
            self.channel_all[channel_idx] = self.normalize_channel(self.channel_all[channel_idx])
        # Bits
        self.tx_pilot_bits_3d, self.tx_uncoded_pilot_bits_2d = self.g_pilot_bits_3d()  # (N_t, num_pilot_symbols, QAM_order*nfft)
        self.tx_data_bits_3d, self.tx_uncoded_data_bits_2d = self.g_data_bits_3d()  # (N_t, num_data_symbols, QAM_order*nfft)

        # Transmitter
        self.tx_pilot_freq_symbols_3d = self.digit_mod.map_symbols(self.tx_pilot_bits_3d)  # (N_t, num_pilot_symbols, nfft)
        self.tx_data_freq_symbols_3d = self.digit_mod.map_symbols(self.tx_data_bits_3d)  # (N_t, num_data_symbols, nfft)

        self.pilot_position = np.arange(self.num_pilot_symbols)
        self.data_position = self.num_pilot_symbols + np.arange(self.num_data_symbols)

        self.tx_pilot_time_3d = self.ofdm_mod.ofdm_mod(
            self.tx_pilot_freq_symbols_3d)  # (N_t, num_pilot_symbols, (nfft+cp))
        self.tx_data_time_3d = self.ofdm_mod.ofdm_mod(
            self.tx_data_freq_symbols_3d)  # (N_t, num_data_symbols, (nfft+cp))
        self.tx_pilot_data_time_3d = np.concatenate([self.tx_pilot_time_3d, self.tx_data_time_3d], axis=1)

        if self.USE_PA:
            self.tx_pilot_time_3d_pa = self.PA.output(self.tx_pilot_time_3d)
            self.tx_data_time_3d_pa = self.PA.output(self.tx_data_time_3d)
        else:
            self.tx_pilot_time_3d_pa = self.tx_pilot_time_3d
            self.tx_data_time_3d_pa = self.tx_data_time_3d
        self.tx_pilot_data_time_3d_pa = np.concatenate([self.tx_pilot_time_3d_pa, self.tx_data_time_3d_pa],
                                                       axis=1)

        # Receiver
        self.rx_pilot_data_time_3d = self.pass_channel_MIMO(self.tx_pilot_data_time_3d_pa,
                                                            channel_idx=channel_idx, subframe_idx=subframe_idx)  # (N_r, total_num_symbols, (nfft+cp))

        self.rx_pilot_time_3d = self.rx_pilot_data_time_3d[:, :self.num_pilot_symbols, :]
        self.rx_data_time_3d = self.rx_pilot_data_time_3d[:, self.num_pilot_symbols:, :]


    def g_pilot_bits_3d(self, num_trans_ant=None, qam_order=None):
        '''Generate pilot bits'''
        if num_trans_ant == None:
            num_trans_ant = self.N_t
        if qam_order == None:
            qam_order = self.QAM_order
        tx_pilot_bits_3d = np.random.random_integers(0, 1, size=[num_trans_ant, self.num_pilot_symbols,
                                                                   qam_order * self.nfft])
        tx_uncoded_pilot_bits_2d = None
        return tx_pilot_bits_3d, tx_uncoded_pilot_bits_2d

    def g_data_bits_3d(self, num_trans_ant=None, qam_order=None):
        '''Generate data bits'''
        if num_trans_ant == None:
            num_trans_ant = self.N_t
        if qam_order == None:
            qam_order = self.QAM_order

        tx_data_bits_3d = np.random.random_integers(0, 1, size = [num_trans_ant, self.num_data_symbols, qam_order*self.nfft])
        tx_uncoded_data_bits_2d = None
        return tx_data_bits_3d, tx_uncoded_data_bits_2d

    def pass_channel_MIMO(self, input, channel_idx=0, subframe_idx=0):
        hx_total = []

        for r in range(self.N_r):
            hx = 0
            for t in range(self.N_t):
                if self.channel_mat_file:
                    h = self.channel_all[channel_idx, t, r, subframe_idx, :, :]
                else:
                    h = self.channel_all[channel_idx, t, r, :, :]
                x = input[t, :, :]
                hx = hx + self.pass_channel(h, x)
            hx_total.append(hx)
        hx_total = np.asarray(hx_total)
        y_total, noise_power = self.add_noise(hx_total)

        return np.asarray(y_total)

    def pass_channel(self, h, x):
        assert (h.shape[0] >= x.shape[0])
        # l = h.shape[1]
        num_symbols = x.shape[0]
        y = np.zeros(x.shape, dtype=complex)
        x = x.reshape(-1)
        x = np.concatenate((np.zeros(self.cp_length), x), axis=0)
        for i in range(num_symbols):
            temp = np.convolve(h[i], x[i * (self.cp_length+self.nfft):(i + 1) * (self.cp_length+self.nfft) + self.cp_length])
            y[i] = temp[self.cp_length:self.cp_length*2+self.nfft]
        return y

    def add_noise(self, d):
        noise_var = self.noise_var
        noise = np.random.normal(0, np.sqrt(noise_var/2), d.shape) + 1j * np.random.normal(0, np.sqrt(
            noise_var/2), d.shape)
        noisy_d = d + noise
        noise_power = np.sum(np.abs(noise) ** 2)
        return noisy_d, noise_power

    def cal_noise_var(self):
        W = 1
        Es = np.mean(np.abs(self.digit_mod.constellation)**2)
        Es = Es/W
        Eb = Es/self.QAM_order # Eb = Es / QAM_order, Es: energy per symbol, QAM_order: bits per symbol
        Eb_rx = Eb
        EbNo_linear = 10**(self.EbNo/10)
        No = Eb_rx/EbNo_linear
        noise_var = No*W
        return noise_var

    def normalize_channel(self, current_channel):
        # current_channel: (n_TX, n_RX, N_OFDM_Sym, len_channel)
        n_OFDM_symbol = current_channel.shape[2]
        h_all_freq = np.fft.fft(current_channel, n=self.nfft, axis = -1)
        pwr = 0
        for j in range(n_OFDM_symbol):
            for m in range(self.nfft):
                h_temp = h_all_freq[:, :, j, m]
                trace_temp = np.trace(h_temp.dot(np.conj(h_temp.T)))
                pwr = pwr + trace_temp
        pwr = pwr / (n_OFDM_symbol * self.nfft * self.N_t)
        normalized_channel = current_channel / np.sqrt(np.abs(pwr))
        return normalized_channel

    def initialize_PA(self):
        W = 15e3
        No_actual = 1e-7
        Eb_actual = No_actual * 10 ** (self.EbNo / 10) * W
        curr_es = np.mean(np.abs(self.digit_mod.symbol_patterns)**2)
        Es_actual = Eb_actual * self.QAM_order / curr_es
        PA = Non_PA(E_s=Es_actual, curr_es=curr_es)
        return PA