import numpy as np

def BER(true_bits, rec_bits):
    if isinstance(true_bits, list) and isinstance(rec_bits, list):
        n_t = len(true_bits)
        error_rate = 0.0
        for i in range(n_t):
            if true_bits[i].shape != rec_bits[i].shape:
                raise Exception('The shape of inputs does not match')
            error_rate += np.sum(true_bits[i] != rec_bits[i]) / true_bits[i].size
        error_rate = error_rate / n_t
        return error_rate
    else:
        if true_bits.shape != rec_bits.shape:
            raise Exception('The shape of inputs does not match')
        return np.sum(true_bits != rec_bits)/true_bits.size