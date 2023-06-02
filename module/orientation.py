import numpy as np
import pandas as pd

class Orientation:
    def __init__(self, idImage):
        self.ori = pd.read_csv(f'../../imagej/orientation/data{idImage:04}.csv')
        self.ori['Orientation'] = self.ori['Orientation']/180.0*np.pi

        print(self.ori)

