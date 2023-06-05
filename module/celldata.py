import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import const
from module.orientation import Orientation

class CellData:
    def __init__(self, idImage):
        self.imgCell  = plt.imread(f'{const.TARGET_DIR}/tif/image{idImage:04}.tif')
        self.imgMask0 = plt.imread(f'{const.TARGET_DIR}/mask/image{idImage:04}-1.tif')

        self.ori = Orientation(idImage)

        print(self.imgCell)
