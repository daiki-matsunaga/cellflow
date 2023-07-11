import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from module import const
from module import utils

class Orientation:
    def __init__(self, idImage, imgMask):
        self.idImage = idImage

        self.df = pd.read_csv(f'{const.DIR}/data/orientations/data{idImage:04}.csv')
        self.df['Orientation'] = self.df['Orientation']/180.0*np.pi # degree to radian

        # rename column name
        self.df.rename(columns={'X': 'x', 'Y': 'y'}, inplace=True)

        # newly added column
        self.df['nx'] = self.df['ny'] = 0.0

        # apply mask to remove orientation outside
        self.df = utils.apply_mask(self.df, imgMask)

        # compute vector
        self.df['nx'] = np.cos(self.df['Orientation'])*self.df['Coherency']
        self.df['ny'] = np.sin(self.df['Orientation'])*self.df['Coherency']

    def draw_orientation(self, imgCell):
        fig = plt.figure(frameon=False)
        plt.imshow(imgCell, cmap="gray")
        plt.quiver(self.df['x'], self.df['y'], self.df['nx'], self.df['ny'],
                   color='y', scale_units='xy', pivot='middle',
                   scale=4.0e-2, width=2.5e-3, headaxislength=0, headlength=0)
        plt.axis("off")

        target_dir = f'{const.DIR}/processed/orientation'
        pathlib.Path(target_dir).mkdir(exist_ok=True)

        fig.savefig(f'{target_dir}/image{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=277.2)
        plt.close()
