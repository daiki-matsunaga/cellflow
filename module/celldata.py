import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import const
from module.orientation import Orientation

class CellData:
    def __init__(self, idImage):
        print(const.UM_PIX)

        self.ori = Orientation(idImage)
