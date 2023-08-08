'''
Copyright (c) 2023, Jiarui Xu, Zhou Zhou, and Lingjia Liu, Virginia Tech.
All rights reserved.
'''

import numpy as np

class Non_PA:
    # Rapp model
    def __init__(self, p = 3, pho0 = 1, E_s = 0.3, curr_es=1, PAPR=None):
        # Parameters setting on the Nonlinearity of PA
        # E_s is the actual average power of input signal
        # The saturated level is normalized as the inverse of signal input power.
        self.A_sat = 1 / np.sqrt(E_s)
        # input backoff(IBO)
        self.IBO = 1 / E_s / curr_es
        if PAPR!=None:
            self.K = self.IBO/PAPR
        # the smooth parameter of PA
        self.p = p
        # the amplification factor
        self.pho0 = pho0

    def output(self, x):
        # x is the input signal of the PA
        v = 1
        A = np.absolute(x)
        A_sat = self.A_sat
        g = (v*x)/np.power((1+np.power((v*A)/A_sat,2*self.p)), 1/(2*self.p))
        y = g*self.pho0
        return y

class PA:
    def __init__(self, p=3, A_sat=1):
        # the smooth parameter of PA
        self.p = p
        # the saturation level
        self.A_sat = A_sat


    def output(self, x):
        x_abs = np.absolute(x)
        y = x/((1+(x_abs/self.A_sat)**(2*self.p))**(1/(2*self.p)))
        return y
