import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from module import utils

class Orientation:
    def __init__(self, config, idImage, imgMask):
        self.config = config
        self.idImage = idImage
        self.imgMask = imgMask

        self.df = pd.read_csv(f'{self.config.DIR}/data/orientations/data{idImage:04}.csv')
        self.df['Orientation'] = self.df['Orientation']/180.0*np.pi # degree to radian

        # rename column name
        self.df.rename(columns={'X': 'x', 'Y': 'y'}, inplace=True)

        # compute pixel difference
        self.oriPixelDiff = self.df.loc[1, 'x'] - self.df.loc[0, 'x'] 
        self.px0 = self.df.loc[0, 'x']
        self.py0 = self.df.loc[0, 'y']

        # newly added column
        self.df['nx'] = self.df['ny'] = 0.0

        # compute vector
        self.df['nx'] = np.cos(self.df['Orientation'])*self.df['Coherency']
        self.df['ny'] = np.sin(self.df['Orientation'])*self.df['Coherency']

        self.calc_defect()

        # apply mask to remove orientation outside
        self.df = utils.apply_mask(self.df, imgMask)
        self.idImage = idImage

    def calc_defect(self):
        self.defect = pd.DataFrame(columns = ['x', 'y', 'charge'])

        # global q tensor
        q = np.zeros((2, 2))

        for index, row in self.df.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            theta = row['Orientation']

            if not utils.isInsideMask(x, y, self.imgMask):
                continue

            tx = np.cos(theta)
            ty = np.sin(theta)
        
            # compute global q tensor
            q[0][0] += tx*tx - 0.5 
            q[0][1] += tx*ty 
            q[1][0] += ty*tx 
            q[1][1] += ty*ty - 0.5 

            angle = 0.0
            angle += self.angle_difference(x, y, +1, +0, +1, +1)
            angle += self.angle_difference(x, y, +1, +1, +0, +1)
            angle += self.angle_difference(x, y, +0, +1, -1, +1)
            angle += self.angle_difference(x, y, -1, +1, -1, +0)
            angle += self.angle_difference(x, y, -1, +0, -1, -1)
            angle += self.angle_difference(x, y, -1, -1, +0, -1)
            angle += self.angle_difference(x, y, +0, -1, +1, -1)
            angle += self.angle_difference(x, y, +1, -1, +1, +0)

            if abs(angle/(2.0*np.pi)) > 1.0e-1:
                tmp = pd.DataFrame([[x, y, -angle/(2.0*np.pi)]], columns=self.defect.columns)
                self.defect = pd.concat([self.defect, tmp], ignore_index=True, axis=0)

        self.defect['isInterpolated'] = False
        for index, row in self.df.iterrows():
            x = int(row['x'])
            y = int(row['y'])

            # extract indecies at four places
            # FIXME: id=0 would be False
            indecies = []
            indecies += filter(None, [utils.get_index_at_position(self.defect, x, y)])
            indecies += filter(None, [utils.get_index_at_position(self.defect, x + self.oriPixelDiff, y)])
            indecies += filter(None, [utils.get_index_at_position(self.defect, x, y + self.oriPixelDiff)])
            indecies += filter(None, [utils.get_index_at_position(self.defect, x + self.oriPixelDiff, y + self.oriPixelDiff)])

            if len(indecies) >= 4:
                local = self.defect.iloc[indecies]
                charges = np.array(local['charge'].tolist())
                
                if all(charges > 0.0) if charges[0] > 0.0 else all(charges < 0.0):
                    average_x = local['x'].mean()
                    average_y = local['y'].mean()

                    tmp = pd.DataFrame([[average_x, average_y, charges.mean(), True]], columns=self.defect.columns)
                    self.defect = pd.concat([self.defect, tmp], ignore_index=True, axis=0)

        self.defect2 = self.defect[self.defect['isInterpolated']]

    def angle_difference(self, x, y, dx1, dy1, dx2, dy2):
        tmp1 = utils.get_index_at_position(self.df, x + dx1*self.oriPixelDiff, y + dy1*self.oriPixelDiff)
        tmp2 = utils.get_index_at_position(self.df, x + dx2*self.oriPixelDiff, y + dy2*self.oriPixelDiff)

        diff = self.df.loc[tmp2, 'Orientation'] - self.df.loc[tmp1, 'Orientation']

        if diff > +np.pi/2.0: diff -= np.pi
        if diff < -np.pi/2.0: diff += np.pi

        return diff

    def draw_orientation(self, imgCell):
        fig = plt.figure(frameon=False)
        plt.imshow(imgCell, cmap="gray")
        plt.quiver(self.df['x'], self.df['y'], self.df['nx'], self.df['ny'],
                   color='y', scale_units='xy', pivot='middle',
                   scale=3.0e-2, width=2.5e-3, headaxislength=0, headlength=0)
        #plt.scatter(self.defect['x'], self.defect['y'], s=0.5, c=self.defect['charge'], cmap='coolwarm', norm=Normalize(vmin=-0.5, vmax=0.5))
        #plt.scatter(self.defect2['x'], self.defect2['y'], s=8, c=self.defect2['charge'], cmap='coolwarm', marker='^', norm=Normalize(vmin=-0.5, vmax=0.5))
        plt.scatter(self.defect2['x'], self.defect2['y'], s=5, c=self.defect2['charge'], cmap='coolwarm', norm=Normalize(vmin=-0.5, vmax=0.5))
        plt.axis('off')

        target_dir = f'{self.config.DIR}/processed/orientation'
        pathlib.Path(target_dir).mkdir(exist_ok=True)

        fig.savefig(f'{target_dir}/image{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=277.2)
        plt.close()
