import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from module import const
from module import utils

target_ux   = 'ux1'
target_uy   = 'uy1'
target_umag = 'mag1'

class Piv:
    def __init__(self, idImage, imgMask):
        self.idImage = idImage

        self.piv = pd.read_csv(f'{const.DIR}/data/piv/result{const.PIV_FRAME_DIFF:02}_{idImage:04}.txt', header=None, delimiter=r'\s+')
        self.piv = self.piv.rename(columns={0: 'x', 1: 'y', 2: 'ux1', 3: 'uy1', 4: 'mag1', 5: 'ang1', 6: 'p1'})
        self.piv = self.piv.rename(columns={7: 'ux2', 8: 'uy2', 9: 'mag2', 10: 'ang2', 11: 'p2'})
        self.piv = self.piv.rename(columns={12: 'ux0', 13: 'uy0', 14: 'mag0', 15: 'flag'})

        # additional columns 
        self.piv['isInsideMask'] = True # 16
        self.piv['vx'] = 0.0; self.piv['vy'] = 0.0; self.piv['vn'] = 0.0 # 17, 18, 19
        self.piv['divergence'] = np.nan # 20

        # multiply to convert from pix/frame to um/min
        coeff = const.UM_PIX/(const.FRAME_INTERVAL*const.PIV_FRAME_DIFF)*60.0
        self.piv[target_ux]   *= coeff
        self.piv[target_uy]   *= coeff
        self.piv[target_umag] *= coeff

        # extract inner part by applying mask
        self.piv = utils.apply_mask(self.piv, imgMask)

        '''
        vmx = self.piv[target_ux].mean()
        vmy = self.piv[target_uy].mean()

        # re-evaluate the velocity by subtracting the average velocity
        for index, row in self.piv.iterrows():
            vx = self.piv.loc[index, target_ux] - vmx
            vy = self.piv.loc[index, target_uy] - vmy

            self.piv.loc[index, 'vx'] = vx
            self.piv.loc[index, 'vy'] = vy
            self.piv.loc[index, 'vn'] = np.sqrt(vx*vx + vy*vy)
        '''

    def draw_flowfield(self, imgCell):
        fig = plt.figure(frameon=False)
        plt.imshow(imgCell, cmap="gray")
        q = plt.quiver(self.piv['x'], self.piv['y'], self.piv[target_ux], -self.piv[target_uy], self.piv[target_umag],
                   cmap='jet', scale=5.0e+0, width=2.5e-3, norm=Normalize(vmin=0.0, vmax=0.2))
        fig.colorbar(q)
        plt.axis("off")
        #plt.show()

        target_dir = f'{const.DIR}/processed/piv'
        pathlib.Path(target_dir).mkdir(exist_ok=True)

        fig.savefig(f'{target_dir}/image{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=203.0)
        plt.close()
