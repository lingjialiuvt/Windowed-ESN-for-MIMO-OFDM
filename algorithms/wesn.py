'''
Copyright (c) 2023, Jiarui Xu, Zhou Zhou, and Lingjia Liu, Virginia Tech.
All rights reserved.
'''

import copy
import numpy as np

class WESN:

    def __init__(self, MIMO_OFDM_para, RC_para, Dataset):

        self.nfft = MIMO_OFDM_para['nfft']
        self.m = MIMO_OFDM_para['QAM_order']
        self.cp_length = MIMO_OFDM_para['CP_length']
        self.Q = MIMO_OFDM_para['training_samples']
        self.T = MIMO_OFDM_para['testing_samples']
        self.N_t = MIMO_OFDM_para['N_t']
        self.N_r = MIMO_OFDM_para['N_r']
        self.EbNo = MIMO_OFDM_para['EbNo']

        self.N_n = RC_para['num_neurons']
        self.sparsity = RC_para['W_tran_sparsity']
        self.spectral_radius = RC_para['W_tran_radius']
        self.max_forget_length = RC_para['max_forget_length']
        self.initial_forget_length = RC_para['initial_forget_length']
        self.forget_length = RC_para['initial_forget_length']
        self.forget_length_search_step = RC_para['forget_length_search_step']
        self.input_scale = RC_para['input_scale']
        self.window_length = RC_para['window_length']

        seed = 10
        self.RS = np.random.RandomState(seed)
        self.N_in = self.N_r * self.window_length * 2
        self.N_out = self.N_t * 2
        self.S_0 = np.zeros([self.N_n])

        self.init_weights()
        self.W_out = self.RS.randn(self.N_out, self.N_n + self.N_in)

        self.digit_mod = Dataset.digit_mod
        self.ofdm_mod = Dataset.ofdm_mod


    def init_weights(self):
        self.W = self.sparse_mat(self.N_n)
        self.W_in = 2 * (self.RS.rand(self.N_n, self.N_in) - 0.5)
        self.W_tran = np.concatenate([self.W, self.W_in], axis=1)

    def sparse_mat(self, m):
        W = self.RS.rand(m, m) - 0.5
        W[self.RS.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (self.spectral_radius / radius)
        return W

    def complex_to_real_target(self, Y_target_2D):
        Y_target_2D_real_list = []
        for t in range(self.N_t):
            target = Y_target_2D[t, :].reshape(1, -1) # (1, N_symbols * (N_fft+N_cp))
            real_target = np.concatenate((np.real(target), np.imag(target)), axis=0)  # (2, N_symbols * (N_fft+N_cp))
            Y_target_2D_real_list.append(real_target)
        Y_target_2D_real = np.concatenate(Y_target_2D_real_list, axis=0)
        return Y_target_2D_real

    def fitting_time(self, Y_3D, Y_target_3D):
        Y_2D = Y_3D.reshape([self.N_r, -1])
        Y_target_2D = Y_target_3D.reshape([self.N_t,-1])

        Y_target_2D = self.complex_to_real_target(Y_target_2D)

        obj_value_delay = []
        W_out_delay = []
        delay_value = []

        for d in np.arange(self.initial_forget_length, self.max_forget_length, self.forget_length_search_step):
            self.forget_length = d # delay
            Y_2D_new = self.form_window_input_signal(Y_2D)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
            S_2D_transit = self.state_transit(Y_2D_new * self.input_scale) # (N_n, N_symbols * (N_fft + N_cp))
            Y_2D_new = Y_2D_new[:, self.forget_length:] # (N_r * window_length, N_symbols * (N_fft + N_cp))
            S_2D = np.concatenate([S_2D_transit, Y_2D_new], axis=0) # (N_n + N_r * window_length, N_symbols * (N_fft + N_cp))
            assert Y_target_2D.shape[0] == self.N_out
            self.W_out = Y_target_2D @ np.linalg.pinv(S_2D)
            obj_value_delay.append(np.sum(np.abs(Y_target_2D - self.W_out @ S_2D)))
            W_out_delay.append(self.W_out)
            delay_value.append(d)
            print('Delay:', d, ', INV: The fitting error is', obj_value_delay[-1])

        indx = np.argmin(obj_value_delay)
        self.forget_length = delay_value[indx]
        self.W_out = W_out_delay[indx]
        print(f'Optimal delay is {self.forget_length}')

    def form_window_input_signal(self, Y_2D_complex):
        # Y_2D: [N_r, N_symbols * (N_fft + N_cp)]
        Y_2D = np.concatenate((Y_2D_complex.real, Y_2D_complex.imag), axis=0)
        Y_2D = np.concatenate([Y_2D, np.zeros([Y_2D.shape[0], self.forget_length], dtype=Y_2D.dtype)], axis=1) # [N_r, N_symbols * (N_fft + N_cp) + delay]
        Y_2D_window = []

        for n in range(self.window_length):
            shift_y_2d = np.roll(Y_2D, shift=n, axis=-1)
            shift_y_2d[:, :n] = 0.
            Y_2D_window.append(shift_y_2d)

        Y_2D_window = np.concatenate(Y_2D_window, axis = 1).reshape(self.N_r * self.window_length * 2, -1)
        return Y_2D_window

    def symbol_detection_time(self, Y_3D):
        Y_2D_org = Y_3D.reshape([self.N_r, -1])

        Y_2D = self.form_window_input_signal(Y_2D_org)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        S_2D = self.state_transit(Y_2D)
        Y_2D = Y_2D[:, self.forget_length:]
        S_2D = np.concatenate([S_2D, Y_2D], axis=0)


        Tx_data_time_symbols_2D = self.W_out @ S_2D
        Tx_data_time_symbols_2D = self.real_to_complex_predict(Tx_data_time_symbols_2D)
        Tx_data_time_symbols_3D = Tx_data_time_symbols_2D.reshape(self.N_t, -1, self.cp_length + self.nfft)

        Tx_data_freq_symbols_3D = self.ofdm_mod.ofdm_demod(Tx_data_time_symbols_3D)

        Tx_data_bits_3D = self.digit_mod.demap_symbols(Tx_data_freq_symbols_3D, 'hard')
        return Tx_data_bits_3D

    def real_to_complex_predict(self, Tx_data_time_symbols_2D):
        predict_complex_list = []
        for t in range(self.N_t):
            curr_complex = Tx_data_time_symbols_2D[t * 2] + 1j * Tx_data_time_symbols_2D[t * 2 + 1]
            predict_complex_list.append(curr_complex.reshape(1, -1))
        predict_complex = np.concatenate(predict_complex_list, axis=0)
        return predict_complex

    def forward(self, Y_2D_complex):
        # Y_2D: [N_r, N_symbols * (N_fft+N_cp)]
        Y_2D = self.form_window_input_signal(Y_2D_complex)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        S_2D = self.state_transit(Y_2D*self.input_scale)
        return S_2D

    def state_transit(self, Y_2D):
        # Y_2D: [N_r, N_symbols * (N_fft+N_cp)]
        T = Y_2D.shape[-1] # Number of samples
        S_1D = copy.deepcopy(self.S_0)
        S_2D = []
        for t in range(T):
            S_1D = np.tanh(self.W_tran @ np.concatenate([S_1D, Y_2D[:, t]], axis=0)) + 1e-6 * (self.RS.rand(self.N_n) - 0.5)
            S_2D.append(S_1D)

        S_2D = np.stack(S_2D, axis=1)
        self.S_0 = S_1D
        return S_2D[:, self.forget_length:]

    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))
