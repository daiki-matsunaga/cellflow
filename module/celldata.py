import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from module.piv import Piv
from module.orientation import Orientation

class CellData:
    def __init__(self, config):
        # initialize lists
        self.intensity_mean = []
        self.average_velocity = []
        self.config = config

        # count image numbers
        self.numImage = len(glob.glob(f'{self.config.DIR}/images/tif/*'))

        self.show_status()

    def show_status(self):
        print('#'*80)
        print(f'target directory: {self.config.DIR}')
        print(f'image number:     {self.numImage}')
        print('#'*80)

    def compute(self, idImage):
        self.idImage = idImage

        # read images
        # Note: mask = 1 (inside), mask = 0 (outside)
        self.imgCell  = plt.imread(f'{self.config.DIR}/images/tif/image{idImage:04}.tif')
        self.imgMask0 = plt.imread(f'{self.config.DIR}/images/mask/image{idImage:04}-1.tif')//255
        self.imgMask1 = plt.imread(f'{self.config.DIR}/images/mask/image{idImage + self.config.PIV_FRAME_DIFF:04}-1.tif')//255

        # erode masks
        kernel = np.ones((5, 5), np.uint8)
        imgMaskA = cv2.erode(self.imgMask0, kernel, iterations=20)
        imgMaskB = cv2.erode(self.imgMask1, kernel, iterations=20)

        # compute velocity field
        if self.config.FLAG_COMPUTE_VELOCITY_FIELD:
            self.piv = Piv(self.config, idImage, (imgMaskA | imgMaskB))
            self.piv.draw_flowfield(self.imgCell)
            self.piv.draw_divergence(self.imgCell)

        # compute orientation field
        if self.config.FLAG_COMPUTE_ORIENTATION_FIELD:
            self.ori = Orientation(self.config, idImage, self.imgMask0)
            self.ori.draw_orientation(self.imgCell)
        
        #self.intensity_mean.append(self.imgCell[self.imgMask0 == 1].mean())
        #self.average_velocity.append(self.piv.average_velocity)

    def draw_figure(self, var, ylabel='y label'):
        fig, ax = plt.subplots()
        ax.plot(var)
        ax.set(xlabel='frame', ylabel=ylabel)
        plt.show()
