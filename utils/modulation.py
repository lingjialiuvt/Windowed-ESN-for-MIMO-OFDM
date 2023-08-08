'''
Copyright (c) 2023, Jiarui Xu, Zhou Zhou, and Lingjia Liu, Virginia Tech.
All rights reserved.
'''

import numpy as np
import commpy as com
from sympy.combinatorics.graycode import GrayCode

class Modulation:

    def __init__(self, m, gray_code, ofdm_structure='OFDM', data_carrier=None, pilot_carrier=None, carrier=None):
        # m is the modulation order
        self.gray_code = gray_code
        self.QAM = com.modulation.QAMModem(2**m)
        self.symbol_patterns = []
        self.m = m
        self.pilots_bit = np.array([])
        constellation_power = 0
        if gray_code:
            gcode = GrayCode(m)
            code_list = list(gcode.generate_gray())
            self.code_list_base2 = [int(i, base=2) for i in code_list]
            symbol_patterns = []
            k = (np.sqrt(2**m) - 2) / 2
            q_symbols = np.arange(-2*k-1, 2*k+2, 2).astype(np.int64)
            i_symbols = q_symbols.copy()
            for i_val in i_symbols:
                for q_val in q_symbols:
                    curr_symbol = i_val + 1j * q_val
                    symbol_patterns.append(curr_symbol)
                    constellation_power += (np.power(abs(curr_symbol), 2)) / (2 ** m)
                q_symbols = np.flip(q_symbols)
            symbol_patterns = np.asarray(symbol_patterns)
            # self.plot_gray_code_const(symbol_patterns)
        else:
            symbol_patterns = np.zeros(2 ** m, dtype='complex')
            for k in range(2**m):
                barray = np.array(list(bin(k)[2:]), dtype=int)
                symbol_patterns[k] = self.QAM.modulate(barray)
                constellation_power += (np.power(abs(symbol_patterns[k]),2))/(2**m)
        # make sure the symbol power is normalized as 1
        # self.constellation_power = 1
        # self.symbol_patterns = symbol_patterns
        self.gray_code = gray_code
        self.constellation_power = constellation_power
        self.symbol_patterns = symbol_patterns/np.sqrt(constellation_power)
        self.constellation = self.symbol_patterns

        self.ofdm_structure = ofdm_structure
        if self.ofdm_structure == 'WiFi_OFDM':
            self.data_carrier = data_carrier
            self.pilot_carrier = pilot_carrier
            self.carrier = carrier

        bit_arr = self.demap_symbols(self.constellation, 'hard').reshape(-1, self.m)
        self.code_list = ["".join(map(str, const_bits)) for const_bits in bit_arr]
        self.bit_arr_base2 = np.asarray([int("".join(map(str, const_bits)), 2) for const_bits in bit_arr])

    def plot_gray_code_const(self, symbol_patterns):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(symbol_patterns.real, symbol_patterns.imag, marker='.')
        for i in range(len(symbol_patterns.real)):
            plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, self.code_list[i])
        plt.show()

    def find_nearby_const_idx(self, curr_point_idx, symbol_patterns):
        if not self.gray_code:
            raise Exception("This function only works when gray code is true.")
        import matplotlib.pyplot as plt
        ele_idx = curr_point_idx
        qam_num = 2 ** self.m
        num_q_comp = np.sqrt(qam_num)
        x_idx = np.floor(ele_idx / num_q_comp)
        y_idx = ele_idx % num_q_comp if x_idx % 2 == 0 else num_q_comp - 1 - ele_idx % num_q_comp
        nearby_const_idx = []
        idx_to_diag_const = []
        if ele_idx != 0: nearby_const_idx.append(ele_idx - 1)
        if ele_idx != qam_num - 1: nearby_const_idx.append(ele_idx + 1)
        if x_idx - 1 >= 0:
            if (x_idx - 1) % 2 == 0:
                left_const_idx = (x_idx - 1) * num_q_comp + y_idx
                if y_idx != num_q_comp - 1:
                    idx_to_diag_const.append(int(left_const_idx + 1))
                if y_idx != 0:
                    idx_to_diag_const.append(int(left_const_idx - 1))
            else:
                left_const_idx = (x_idx - 1) * num_q_comp + num_q_comp - 1 - y_idx
                if y_idx != num_q_comp - 1:
                    idx_to_diag_const.append(int(left_const_idx - 1))
                if y_idx != 0:
                    idx_to_diag_const.append(int(left_const_idx + 1))
            nearby_const_idx.append(int(left_const_idx))
        if x_idx + 1 <= num_q_comp - 1:
            if (x_idx + 1) % 2 == 0:
                right_const_idx = (x_idx + 1) * num_q_comp + y_idx
                if y_idx != num_q_comp - 1:
                    idx_to_diag_const.append(int(right_const_idx + 1))
                if y_idx != 0:
                    idx_to_diag_const.append(int(right_const_idx - 1))
            else:
                right_const_idx = (x_idx + 1) * num_q_comp + num_q_comp - 1 - y_idx
                if y_idx != num_q_comp - 1:
                    idx_to_diag_const.append(int(right_const_idx - 1))
                if y_idx != 0:
                    idx_to_diag_const.append(int(right_const_idx + 1))
            nearby_const_idx.append(int(right_const_idx))
        plt.figure()
        plt.scatter(symbol_patterns.real, symbol_patterns.imag, marker='.')
        plt.scatter(symbol_patterns.real[ele_idx], symbol_patterns.imag[ele_idx], marker='o')
        for i in range(len(symbol_patterns.real)):
            # plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, self.code_list[i])
            if symbol_patterns.imag[i] > 0:
                curr_text = "%d+%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            else:
                curr_text = "%d%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, curr_text)
        for idx in nearby_const_idx:
            plt.scatter(symbol_patterns.real[idx], symbol_patterns.imag[idx], marker='x', color='r')
        for idx in idx_to_diag_const:
            plt.scatter(symbol_patterns.real[idx], symbol_patterns.imag[idx], marker='x', color='g')
        plt.show()
        return nearby_const_idx

    def find_same_real_or_imag(self, curr_point_idx, symbol_patterns):
        if not self.gray_code:
            raise Exception("This function only works when gray code is true.")
        import matplotlib.pyplot as plt
        ele_idx = curr_point_idx
        curr_const = symbol_patterns[ele_idx]
        qam_num = 2 ** self.m
        num_q_comp = np.sqrt(qam_num)
        x_idx = np.floor(ele_idx / num_q_comp)
        y_idx = ele_idx % num_q_comp if x_idx % 2 == 0 else num_q_comp - 1 - ele_idx % num_q_comp
        same_real_or_image_idx = []
        for idx, const in enumerate(symbol_patterns):
            if (const.real == curr_const.real or const.imag == curr_const.imag) and idx != ele_idx:
                same_real_or_image_idx.append(idx)
        plt.figure()
        plt.scatter(symbol_patterns.real, symbol_patterns.imag, marker='.')
        plt.scatter(symbol_patterns.real[ele_idx], symbol_patterns.imag[ele_idx], marker='o')
        for i in range(len(symbol_patterns.real)):
            # plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, self.code_list[i])
            if symbol_patterns.imag[i] > 0:
                curr_text = "%d+%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            else:
                curr_text = "%d%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            if i == ele_idx: curr_text = f"{curr_text}(GT)"
            plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, curr_text)
        for idx in same_real_or_image_idx:
            plt.scatter(symbol_patterns.real[idx], symbol_patterns.imag[idx], marker='x', color='r')
        plt.show()
        return same_real_or_image_idx

    def find_up_shift_const(self, curr_point_idx, symbol_patterns):
        if not self.gray_code:
            raise Exception("This function only works when gray code is true.")
        import matplotlib.pyplot as plt
        ele_idx = curr_point_idx
        idx_to_up_const = []
        qam_num = 2 ** self.m
        num_q_comp = np.sqrt(qam_num)
        x_idx = np.floor(ele_idx / num_q_comp)
        if x_idx % 2 == 0:
            y_idx = ele_idx % num_q_comp
            if y_idx != num_q_comp - 1:
                idx_to_up_const.append(ele_idx + 1)
            else:
                idx_to_up_const.append(ele_idx)
        else:
            y_idx = num_q_comp - 1 - ele_idx % num_q_comp
            if y_idx != num_q_comp - 1:
                idx_to_up_const.append(ele_idx - 1)
            else:
                idx_to_up_const.append(ele_idx)
        plt.figure()
        plt.scatter(symbol_patterns.real, symbol_patterns.imag, marker='.')
        plt.scatter(symbol_patterns.real[ele_idx], symbol_patterns.imag[ele_idx], marker='o')
        for i in range(len(symbol_patterns.real)):
            # plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, self.code_list[i])
            if symbol_patterns.imag[i] > 0:
                curr_text = "%d+%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            else:
                curr_text = "%d%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            if i == ele_idx: curr_text = f"{curr_text}(GT)"
            plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, curr_text)
        for idx in idx_to_up_const:
            plt.scatter(symbol_patterns.real[idx], symbol_patterns.imag[idx], marker='x', color='r')
        plt.show()
        return idx_to_up_const

    def find_up_and_down_shift_const(self, curr_point_idx, symbol_patterns):
        if not self.gray_code:
            raise Exception("This function only works when gray code is true.")
        import matplotlib.pyplot as plt
        ele_idx = curr_point_idx
        idx_to_up_down_const = []
        qam_num = 2 ** self.m
        num_q_comp = np.sqrt(qam_num)
        x_idx = np.floor(ele_idx / num_q_comp)
        if x_idx % 2 == 0:
            y_idx = ele_idx % num_q_comp
            if y_idx != num_q_comp - 1 and y_idx != 0:
                idx_to_up_down_const.append([ele_idx + 1, ele_idx - 1])
            else:
                idx_to_up_down_const.append([ele_idx])
        else:
            y_idx = num_q_comp - 1 - ele_idx % num_q_comp
            if y_idx != num_q_comp - 1 and y_idx != 0:
                idx_to_up_down_const.append([ele_idx + 1, ele_idx - 1])
            else:
                idx_to_up_down_const.append([ele_idx])
        plt.figure()
        plt.scatter(symbol_patterns.real, symbol_patterns.imag, marker='.')
        plt.scatter(symbol_patterns.real[ele_idx], symbol_patterns.imag[ele_idx], marker='o')
        for i in range(len(symbol_patterns.real)):
            # plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, self.code_list[i])
            if symbol_patterns.imag[i] > 0:
                curr_text = "%d+%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            else:
                curr_text = "%d%dj" % (symbol_patterns.real[i], symbol_patterns.imag[i])
            if i == ele_idx: curr_text = f"{curr_text}(GT)"
            plt.text(symbol_patterns.real[i], symbol_patterns.imag[i] + 0.1, curr_text)
        for up_down_const_list in idx_to_up_down_const:
            for idx in up_down_const_list:
                plt.scatter(symbol_patterns.real[idx], symbol_patterns.imag[idx], marker='x', color='r')
        plt.show()
        return idx_to_up_down_const

    def map_symbols(self, bits):
        symbols_index = self.binary2dec(bits)
        symbols = self.constellation[symbols_index]
        return symbols

    def binary2dec(self, x):
        base = 2**np.arange(self.m) # unpackbits using 'little'
        array_shape2 = x.shape[:-1] + (-1, self.m)
        x2 = x.reshape(array_shape2)
        y = np.dot(x2, base)
        if self.gray_code:
            index = y.copy()
            for i, val in enumerate(self.code_list_base2):
                y[index == val] = i
        return y

    def demap_symbols(self, rx_symbols, demod_type, noise_var=0, type='data'):
        if self.ofdm_structure == 'WiFi_OFDM':
            if type == 'data':
                rx_symbols = rx_symbols[..., self.data_carrier]
            elif type == 'pilot':
                rx_symbols = rx_symbols[..., self.pilot_carrier]
            elif type == 'sync':
                rx_symbols = rx_symbols[..., self.carrier]
            elif type == 'raw':
                rx_symbols = rx_symbols
            else:
                raise Exception("Type must be specified when using WiFi_OFDM.")

        out_array_shape = rx_symbols.shape[:-1] + (-1,)
        if demod_type == 'hard':
            d_symbols_func = np.vectorize(self.constell2ints)
            d_ints = d_symbols_func(rx_symbols)
            # gray coding
            if self.gray_code:
                d_ints_old = d_ints.copy()
                for i, val in enumerate(self.code_list_base2):
                    d_ints[d_ints_old == i] = val
            d_bits = np.unpackbits(np.expand_dims(d_ints, axis = -1), axis = -1, count=self.m, bitorder='little').reshape(out_array_shape)
        elif demod_type == 'soft':
            # noise_var = noise_var / (1 + noise_var)
            # constellation = self.constellation[self.bit_arr_base2.argsort()]
            # table = np.zeros([2 ** self.m, self.m], dtype=int)
            # for k in range(2 ** self.m): # or for k in self.bit_arr_base2
            #     binary_list = list(bin(k)[2:])
            #     table[k, -len(binary_list):] = np.array(binary_list, dtype=int)
            # indx_0 = np.zeros([2 ** self.m // 2, self.m], dtype=int)
            # indx_1 = np.zeros([2 ** self.m // 2, self.m], dtype=int)
            # for k in range(self.m):
            #     indx_0[:, k] = np.where(table[:, k] == 0)[0]
            #     indx_1[:, k] = np.where(table[:, k] == 1)[0]
            #
            # input_symbols = rx_symbols.reshape(-1)
            # demod_bits = np.zeros(len(input_symbols) * self.m)
            # for n_syms in np.arange(len(input_symbols)):
            #     temp_l = np.zeros(self.m)
            #     prob = np.exp(-abs(input_symbols[n_syms] - constellation) ** 2 / (2*noise_var))
            #     for k in range(self.m):
            #         temp_l[k] = np.log(np.sum(prob[indx_1[:, k]])) - np.log(np.sum(prob[indx_0[:, k]]))
            #     demod_bits[n_syms * self.m:(n_syms+1) * self.m] = temp_l
            #
            # demod_bits[demod_bits == -np.inf] = -500
            # demod_bits[demod_bits == np.inf] = 500
            # d_bits = demod_bits.reshape(out_array_shape)

            constellation = self.constellation[self.bit_arr_base2.argsort()]
            input_symbols = rx_symbols.reshape(-1)
            demod_bits = np.zeros(len(input_symbols) * self.m)
            for i in np.arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in np.arange(self.m):
                    llr_num = 0
                    llr_den = 0
                    for bit_value, symbol in enumerate(constellation):
                        if (bit_value >> bit_index) & 1:
                            llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / (2*noise_var))
                        else:
                            llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / (2*noise_var))
                    demod_bits[i * self.m + self.m - 1 - bit_index] = np.log(llr_num) - np.log(llr_den)
            demod_bits[demod_bits == -np.inf] = -500
            demod_bits[demod_bits == np.inf] = 500
            d_bits = demod_bits.reshape(out_array_shape)
        else:
            raise ValueError('demod_type must be "hard" or "soft"')
        return d_bits

    def constell2ints(self, symbol):
        indx = np.argmin(np.abs(symbol - self.constellation)).astype(np.uint8)
        return indx

    def nearest_neighbor_symbols(self, rx_symbols):
        d_symbols_func = np.vectorize(self.constell2ints)
        d_ints = d_symbols_func(rx_symbols)
        output_symbols = np.zeros(rx_symbols.shape, dtype=rx_symbols.dtype)
        for i, val in enumerate(self.constellation):
            output_symbols[d_ints == i] = val
        return output_symbols

class OFDM_Modulation:

    def __init__(self, nfft, cp_length):
        self.nfft = nfft
        self.cp_length = cp_length

    def ofdm_mod(self, ofdm_sym_freq_org, shift=True, add_cp = True):
        if shift:
            ofdm_sym_freq = np.concatenate((ofdm_sym_freq_org[..., int(self.nfft / 2):],
                                            ofdm_sym_freq_org[..., :int(self.nfft / 2)]),
                                            axis=-1)  # this align with gnuradio 'shifted' option
        else:
            ofdm_sym_freq = ofdm_sym_freq_org
        ofdm_sym_time = np.fft.ifft(ofdm_sym_freq, norm='ortho', axis = -1)
        if add_cp:
            curr_cp = ofdm_sym_time[..., -self.cp_length:]
            ofdm_sym_time = np.concatenate([curr_cp, ofdm_sym_time], axis = -1)
        return ofdm_sym_time

    def ofdm_demod(self, ofdm_symbols, shift=True, remove_cp = True):
        # input is 3D
        # remove cp
        if remove_cp:
            assert ofdm_symbols.shape[-1] == self.nfft + self.cp_length
            ofdm_symbols = ofdm_symbols[..., -self.nfft:]
        freq_ofdm_symbols = np.fft.fft(ofdm_symbols, norm='ortho', axis = -1)
        if shift:
            freq_ofdm_symbols = np.concatenate((freq_ofdm_symbols[..., int(self.nfft / 2):],
                                        freq_ofdm_symbols[..., :int(self.nfft / 2)]),
                                        axis=-1)  # this align with gnuradio 'shifted' option
        return freq_ofdm_symbols
