import numpy as np
import pandas as pd
from module import const

class Orientation:
    def __init__(self, idImage):
        self.ori = pd.read_csv(f'{const.DIR}/orientations/data{idImage:04}.csv')
        self.ori['Orientation'] = self.ori['Orientation']/180.0*np.pi # degree to radian

        # newly added column
        self.ori['nx'] = self.ori['ny'] = 0.0

        #print(self.ori)

    def is_inside_mask(self):
        for index, row in df.iterrows():
            x = int(row['X'])
            y = int(row['Y'])

            if x >= self.imgMask0.shape[1] or y >= self.imgMask0.shape[0]:
                df.loc[index, 'isInsideMask'] = False
                continue

            if self.imgMask0[y][x] == 0:
                df.loc[index, 'isInsideMask'] = False
                continue

            theta = row["Orientation"]
            coherency = row["Coherency"]
            tx = np.cos(theta)
            ty = np.sin(theta)
            df.loc[index, 'nx'] = coherency*tx
            df.loc[index, 'ny'] = coherency*ty
