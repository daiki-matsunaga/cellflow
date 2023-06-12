import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from module import const
from module.piv import Piv
from module.orientation import Orientation

class CellData:
    def __init__(self):
        # initialize lists
        self.intensity_mean = []

        # count image numbers
        self.numImage = len(glob.glob(f'{const.DIR}/images/tif/*'))

        self.show_status()

    def show_status(self):
        print('#'*80)
        print(f'target directory: {const.DIR}')
        print(f'image number:     {self.numImage}')
        print('#'*80)

    def read(self, idImage):
        self.idImage = idImage

        # read images
        # Note: mask = 1 (inside), mask = 0 (outside)
        self.imgCell  = plt.imread(f'{const.DIR}/images/tif/image{idImage:04}.tif')
        self.imgMask0 = plt.imread(f'{const.DIR}/images/mask/image{idImage:04}-1.tif')//255
        self.imgMask1 = plt.imread(f'{const.DIR}/images/mask/image{idImage + const.PIV_FRAME_DIFF:04}-1.tif')//255

        # erode masks
        kernel = np.ones((5, 5), np.uint8)
        imgMaskA = cv2.erode(self.imgMask0, kernel, iterations=20)
        imgMaskB = cv2.erode(self.imgMask1, kernel, iterations=20)

        # read data
        self.piv = Piv(idImage, (imgMaskA | imgMaskB))
        self.ori = Orientation(idImage)
        
        self.intensity_mean.append(self.imgCell[self.imgMask0 == 1].mean())

    def analyse(self):
        self.piv.draw_flowfield(idImage, self.imgCell)

    def draw_figure(self, var, ylabel='y label'):
        fig, ax = plt.subplots()
        ax.plot(var)
        ax.set(xlabel='frame', ylabel=ylabel)
        plt.show()
