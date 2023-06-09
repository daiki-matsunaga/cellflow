import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from module import const

class Piv:
    def __init__(self, idImage, imgCell, imgMask):
        self.piv = pd.read_csv(f'{const.DIR}/data/piv/result{const.PIV_FRAME_DIFF:02}_{idImage:04}.txt', header=None, delimiter=r'\s+')
        self.piv = self.piv.rename(columns={0: 'x', 1: 'y', 2: 'ux1', 3: 'uy1', 4: 'mag1', 5: 'ang1', 6: 'p1'})
        self.piv = self.piv.rename(columns={7: 'ux2', 8: 'uy2', 9: 'mag2', 10: 'ang2', 11: 'p2'})
        self.piv = self.piv.rename(columns={12: 'ux0', 13: 'uy0', 14: 'mag0', 15: 'flag'})
        self.piv['isInsideMask'] = True # 16

        # extract inner part (set 'isInsideMask' to False if outside)
        for index, row in self.piv.iterrows():
            x = int(row['x'])
            y = int(row['y'])

            if imgMask[y][x] == 0:
                self.piv.loc[index, 'isInsideMask'] = False

        self.piv = self.piv[self.piv['isInsideMask']]

        # multiply to convert from pix/frame to um/min
        coeff = const.UM_PIX/(const.FRAME_INTERVAL*const.PIV_FRAME_DIFF)*60.0
        self.piv.ux0  *= coeff
        self.piv.uy0  *= coeff
        self.piv.mag0 *= coeff

        self.draw_flowfield(idImage, imgCell)

    def draw_flowfield(self, idImage, imgCell):
        fig = plt.figure(frameon=False)
        plt.imshow(imgCell, cmap="gray")
        q = plt.quiver(self.piv['x'], self.piv['y'], self.piv['ux0'], - self.piv['uy0'], self.piv['mag0'],
                   cmap='jet', scale=50.0e+0, width=2.5e-3, norm=Normalize(vmin=0.0, vmax=2.0))
        fig.colorbar(q)
        plt.axis("off")
        #plt.show()

        target_dir = f'{const.DIR}/processed/piv'
        pathlib.Path(target_dir).mkdir(exist_ok=True)

        fig.savefig(f'{target_dir}/image{idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=203.0)
        plt.close()

        '''
        df['vx'] = 0.0; df['vy'] = 0.0; df['vn'] = 0.0 # 17, 18, 19
        df['divergence'] = np.nan # 20

        # erode to extract central part
        kernel = np.ones((5,5), np.uint8)
        imgMaskA = cv2.erode(self.imgMask1, kernel, iterations=20)
        imgMaskB = cv2.erode(self.imgMask2, kernel, iterations=20)

        # extract central part (set 'isInsideMask' to False if outside)
        for index, row in df.iterrows():
            x = int(row['x'])
            y = int(row['y'])

            if imgMaskA[y][x] == 0 or imgMaskB[y][x] == 0:
                df.loc[index, 'isInsideMask'] = False

        # build new data frame
        df2 = df[df['isInsideMask']]
        vmx = df2['ux1'].mean()
        vmy = df2['uy1'].mean()

        vm.append(df2['vn'].mean())

        return df2['vn'].tolist(), evalue.max(), df['divergence'].tolist()
        '''

