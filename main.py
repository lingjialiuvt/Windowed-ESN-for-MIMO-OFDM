'''
Copyright (c) 2023, Jiarui Xu, Zhou Zhou, and Lingjia Liu, Virginia Tech.
All rights reserved.
'''

import numpy as np
from utils.signal_generation import MIMO_OFDM_signal
from algorithms.wesn import WESN
from utils.utils import BER

np.random.seed(0)

MIMO_OFDM_para = {
    'training_samples': 4, # Number of pilot symbols
    'testing_samples': 13-4, # Number of data symbols
    'CP_length': 160, # Cyclic prefix length
    'N_r': 4, # Number of received antennas
    'N_t': 4, # Number of transmit antennas
    'nfft': 1024, # Number of subcarriers
    'QAM_order': 4, # Modulation order, 2^m - QAM
    'EbNo':20,
    'channel_path': 'channels/channel_all.npy', # Channel Path
    'gray_code': True, # Whether use gray coding
    'USE_PA': False, # Whether use power amplifier or not
}

RC_para_standard = \
    {
    'num_neurons': 16, # Number of neurons
    'W_tran_sparsity': 0.4, # Sparsity of the reservoir weights
    'W_tran_radius': 0.5, # Spectral radius of the reservoir weights
    'input_scale': 0.8, # Scale the amplitude of the input
    'initial_forget_length': 40, # Delay searching start value
    'max_forget_length': 80, # Delay searching end value
    'forget_length_search_step': 5, # Delay searching step size
    'window_length': 128, # The buffer length of WESN
    }

# initialize class
subframe_idx = 0

EbNo_list = np.arange(0, 30 + 1, 5)
channel_num = 10

BER_WESN = np.zeros([len(EbNo_list), channel_num])

for eb_idx, EbNo in enumerate(EbNo_list):
    MIMO_OFDM_para['EbNo'] = EbNo
    for channel_idx in range(channel_num):
        mimo_ofdm_dataset = MIMO_OFDM_signal(MIMO_OFDM_para)
        mimo_ofdm_dataset.generate_data(channel_idx=channel_idx,
                                        subframe_idx=subframe_idx)

        rc_detector = WESN(MIMO_OFDM_para, RC_para_standard, mimo_ofdm_dataset)
        rc_detector.fitting_time(mimo_ofdm_dataset.rx_pilot_time_3d, mimo_ofdm_dataset.tx_pilot_time_3d)
        detect_rx_pilot_bits_3d = rc_detector.symbol_detection_time(mimo_ofdm_dataset.rx_pilot_time_3d)
        print('Training BER of WESN is:', BER(detect_rx_pilot_bits_3d, mimo_ofdm_dataset.tx_pilot_bits_3d))
        detect_rx_data_bits_3d = rc_detector.symbol_detection_time(mimo_ofdm_dataset.rx_data_time_3d)
        BER_WESN[eb_idx, channel_idx] = BER(detect_rx_data_bits_3d, mimo_ofdm_dataset.tx_data_bits_3d)
        print('Testing BER of WESN is:', BER_WESN[eb_idx, channel_idx])

print(f"Tested Eb/No: {EbNo_list}, Use Power Amplifier: {MIMO_OFDM_para['USE_PA']}")
print(f"Averaged BER of WESN: {np.mean(BER_WESN, axis = 1)}")