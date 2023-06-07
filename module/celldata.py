import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from module import const
from module.orientation import Orientation

class CellData:
    def __init__(self):
        # initialize lists
        self.intensity_mean = []

        # count image numbers
        self.numImage = len(glob.glob(const.DIR + 'tif/*'))

        self.show_status()

    def show_status(self):
        print('#'*80)
        print(f'target directory: {const.DIR}')
        print(f'image number:     {self.numImage}')
        print('#'*80)

    def read(self, idImage):
        self.imgCell = plt.imread(f'{const.DIR}/tif/image{idImage:04}.tif')

        # mask = 1: inside, mask = 0: outside
        self.imgMask0 = plt.imread(f'{const.DIR}/mask/image{idImage:04}-1.tif')//255

        self.ori = Orientation(idImage)

        self.intensity_mean.append(self.imgCell[self.imgMask0 == 1].mean())

    def draw_figure(self, var, ylabel='y label'):
        fig, ax = plt.subplots()
        ax.plot(var)
        ax.set(xlabel='frame', ylabel=ylabel)
        plt.show()
