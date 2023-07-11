import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from module import const
from module import utils
from module.const import TARGET_U

import sys

class Piv:
    def __init__(self, idImage, imgMask):
        self.idImage = idImage

        # read and set column names
        # see here for details: https://sites.google.com/site/qingzongtseng/piv/tuto?authuser=0
        self.df = pd.read_csv(f'{const.DIR}/data/piv/result{const.PIV_FRAME_DIFF:02}_{idImage:04}.txt', header=None, delimiter=r'\s+')
        self.df = self.df.rename(columns={0: 'x', 1: 'y', 2: 'ux1', 3: 'uy1', 4: 'mag1', 5: 'ang1', 6: 'p1'})
        self.df = self.df.rename(columns={7: 'ux2', 8: 'uy2', 9: 'mag2', 10: 'ang2', 11: 'p2'})
        self.df = self.df.rename(columns={12: 'ux0', 13: 'uy0', 14: 'mag0', 15: 'flag'})

        # multiply to convert from pix/frame to um/min
        coeff = const.UM_PIX/(const.FRAME_INTERVAL*const.PIV_FRAME_DIFF)*60.0
        self.df['vx'] = self.df[TARGET_U['x']]*coeff
        self.df['vy'] = self.df[TARGET_U['y']]*coeff
        self.df['vn'] = self.df[TARGET_U['mag']]*coeff

        # extract inner part by applying mask
        # df_mask: dataframe that contains data inside the mask 
        self.df_mask = utils.apply_mask(self.df, imgMask)

        # subtract by average velocity if True
        if const.FLAG_SUBTRACT_AVERAGE_PIV:
            vmx = self.df_mask['vx'].mean()
            vmy = self.df_mask['vy'].mean()

            # re-evaluate the velocity by subtracting the average velocity
            for index, row in self.df_mask.iterrows():
                vx = self.df_mask.loc[index, 'vx'] - vmx
                vy = self.df_mask.loc[index, 'vy'] - vmy

                self.df_mask.loc[index, 'vx'] = vx
                self.df_mask.loc[index, 'vy'] = vy
                self.df_mask.loc[index, 'vn'] = np.sqrt(vx*vx + vy*vy)

        self.average_velocity = self.df_mask['vn'].mean()

        # calculate divergence
        self.calc_divergence()

    def calc_divergence(self):
        # compute pixel number between plots
        pivPixelDiff = self.df.loc[1, 'x'] - self.df.loc[0, 'x'] 

        self.df['divergence'] = np.nan # 20
        for index, row in self.df_mask.iterrows():
            if not row['isInsideMask']: continue

            x = int(row['x'])
            y = int(row['y'])
            u = row['vx'] 
            v = row['vy'] 

            # check value existance of neighbours
            # Note: dataframe is df_mask to only use the inner part data
            fxp = self.df_mask.index[((self.df_mask['x'] == x + pivPixelDiff) & (self.df_mask['y'] == y))].tolist()
            fxm = self.df_mask.index[((self.df_mask['x'] == x - pivPixelDiff) & (self.df_mask['y'] == y))].tolist()
            fyp = self.df_mask.index[((self.df_mask['x'] == x) & (self.df_mask['y'] == y + pivPixelDiff))].tolist()
            fym = self.df_mask.index[((self.df_mask['x'] == x) & (self.df_mask['y'] == y - pivPixelDiff))].tolist()

            # if the value exist, return index
            # if the value does not exist, return False
            fxp = False if len(fxp) == 0 else fxp[0]
            fxm = False if len(fxm) == 0 else fxm[0]
            fyp = False if len(fyp) == 0 else fyp[0]
            fym = False if len(fym) == 0 else fym[0]

            if fxp and fxm:
                up = self.df.loc[fxp, 'vx']
                um = self.df.loc[fxm, 'vx']
                rurx = (up - um)/(2.0*pivPixelDiff*const.UM_PIX)
            else:
                rurx = np.nan
            """
            elif fxp:
                up = self.df.loc[fxp, 'vx']
                rurx = (up - u)/(pivPixelDiff*const.UM_PIX)
            elif fxm:
                um = self.df.loc[fxm, 'vx']
                rurx = (u - um)/(pivPixelDiff*const.UM_PIX)
            """

            if fyp and fym:
                vp = self.df.loc[fyp, 'vy']
                vm = self.df.loc[fym, 'vy']
                rvry = (vp - vm)/(2.0*pivPixelDiff*const.UM_PIX)
            else:
                rvry = np.nan
            """
            elif fyp:
                vp = self.df.loc[fyp, 'vy']
                rvry = (vp - v)/(pivPixelDiff*const.UM_PIX)
            elif fym:
                vm = self.df.loc[fym, 'vy']
                rvry = (v - vm)/(pivPixelDiff*const.UM_PIX)
            """

            self.df.loc[index, 'divergence'] = rurx + rvry

    def draw_flowfield(self, imgCell):
        fig = plt.figure(frameon=False)
        plt.imshow(imgCell, cmap="gray")
        q = plt.quiver(self.df_mask['x'], self.df_mask['y'], self.df_mask['vx'], -self.df_mask['vy'], self.df_mask['vn'],
                   cmap='jet', scale=5.0e+0, width=2.5e-3, norm=Normalize(vmin=0.0, vmax=0.2))
        fig.colorbar(q)
        plt.axis("off")
        #plt.show()

        target_dir = f'{const.DIR}/processed/piv'
        pathlib.Path(target_dir).mkdir(exist_ok=True)

        fig.savefig(f'{target_dir}/image{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=203.0)
        plt.close()

    def draw_divergence(self, imgCell):
        X = np.array(self.df['x']).reshape(62, 62)
        Y = np.array(self.df['y']).reshape(62, 62)
        D = np.array(self.df['divergence']).reshape(62, 62)
        vmin = -5.0e-3
        vmax = +5.0e-3
        levels = np.linspace(vmin, vmax, 51)

        fig = plt.figure(frameon=False)
        plt.imshow(imgCell, cmap="gray")
        #plt.scatter(df['x'], df['y'], s=1, c=df['divergence'], norm=Normalize(vmin=-5.0e-3, vmax=5.0e-3))
        c = plt.contourf(X, Y, D, levels=levels, cmap='coolwarm', alpha=.2, extend='both', antialiased=True)
        cbar = fig.colorbar(c, ticks=[vmin, vmin/2.0, 0.0, vmax/2.0, vmax])
        cbar.solids.set(alpha=1)

        plt.axis("off")

        target_dir = f'{const.DIR}/processed/divergence'
        pathlib.Path(target_dir).mkdir(exist_ok=True)

        #plt.show()
        fig.savefig(f'{target_dir}/image{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=208.0)
        plt.close()
