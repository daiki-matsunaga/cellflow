import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 

class Data:
    def __init__(self, idImage):
        # settings
        um_pix = 1.0/4.8272
        self.pivFrameDiff = 4
        self.idImage = idImage
        self.idTarget = idImage + (int)(self.pivFrameDiff/2)
        self.um_pix = um_pix

        # read images
        self.imgCell  = plt.imread(f'../../images/tif/image{self.idTarget:04}.tif')
        self.imgMask0 = plt.imread(f'../../images/mask/image{self.idTarget:04}-1.tif')
        self.imgMask1 = plt.imread(f'../../images/mask/image{idImage:04}-1.tif')
        self.imgMask2 = plt.imread(f'../../images/mask/image{idImage + self.pivFrameDiff:04}-1.tif')

    def Piv(self, vm):
        # constants for piv
        interval = 45.0*self.pivFrameDiff 
        coeff = self.um_pix/interval*60.0

        # read piv results
        df = pd.read_csv(f'../../imagej/piv/result{self.pivFrameDiff:02}_{self.idImage:04}.txt', header=None, delimiter=r'\s+')
        df = df.rename(columns={0: 'x', 1: 'y', 2: 'ux1', 3: 'uy1', 4: 'mag1', 5: 'ang1', 6: 'p1'})
        df = df.rename(columns={7: 'ux2', 8: 'uy2', 9: 'mag2', 10: 'ang2', 11: 'p2'})
        df = df.rename(columns={12: 'ux0', 13: 'uy0', 14: 'mag0', 15: 'flag'})
        df['isInsideMask'] = True # 16
        df['vx'] = 0.0; df['vy'] = 0.0; df['vn'] = 0.0 # 17, 18, 19
        df['divergence'] = np.nan # 20

        # multiply to convert from pix/frame to um/min
        df.ux0  *= coeff
        df.uy0  *= coeff
        df.mag0 *= coeff

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

        # q tensor
        qv = np.zeros((2, 2))

        # re-evaluate the velocity by subtracting the average velocity
        for index, row in df2.iterrows():
            vx = df2.loc[index, 'ux1'] - vmx 
            vy = df2.loc[index, 'uy1'] - vmy 

            df2.loc[index, 'vx'] = vx
            df2.loc[index, 'vy'] = vy
            df2.loc[index, 'vn'] = np.sqrt(vx*vx + vy*vy) 

            qv += self.Q_tensor(np.arctan2(vy, vx)) 

        evalue, evec = np.linalg.eig(2.0*qv/len(df2))

        fig = plt.figure(frameon=False)
        plt.imshow(self.imgCell, cmap="gray")
        q = plt.quiver(df2['x'], df2['y'], df2['vx'], - df2['vy'], df2['vn'],
                   cmap='jet', scale=50.0e+0, width=2.5e-3, norm=Normalize(vmin=0.0, vmax=2.0))
        fig.colorbar(q)
        plt.axis("off")

        #plt.show()
        fig.savefig(f'result/image{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=203.0)
        plt.close()

        vm.append(df2['vn'].mean())
        self.PivDivergence(df, df2)

        return df2['vn'].tolist(), evalue.max(), df['divergence'].tolist()

    def PivDivergence(self, df, df2):
        pivPixelDiff = 16

        for index, row in df2.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            u = row['vx'] 
            v = row['vy'] 

            if not row['isInsideMask']: continue

            # check value existance of neighbours
            fxp = ((df2['x'] == x + pivPixelDiff) & (df2['y'] == y)).any()
            fxm = ((df2['x'] == x - pivPixelDiff) & (df2['y'] == y)).any()
            fyp = ((df2['x'] == x) & (df2['y'] == y + pivPixelDiff)).any()
            fym = ((df2['x'] == x) & (df2['y'] == y - pivPixelDiff)).any()

            if fxp and fxm:
                up = df2[(df2['x'] == x + pivPixelDiff) & (df2['y'] == y)].iloc[0, 17]
                um = df2[(df2['x'] == x - pivPixelDiff) & (df2['y'] == y)].iloc[0, 17]
                rurx = (up - um)/(2.0*pivPixelDiff*self.um_pix)
            elif fxp:
                up = df2[(df2['x'] == x + pivPixelDiff) & (df2['y'] == y)].iloc[0, 17]
                rurx = (up - u)/(pivPixelDiff*self.um_pix)
            elif fxm:
                um = df2[(df2['x'] == x - pivPixelDiff) & (df2['y'] == y)].iloc[0, 17]
                rurx = (u - um)/(pivPixelDiff*self.um_pix)
            else:
                rurx = np.nan

            if fyp and fym:
                vp = df2[(df2['x'] == x) & (df2['y'] == y + pivPixelDiff)].iloc[0, 18]
                vm = df2[(df2['x'] == x) & (df2['y'] == y - pivPixelDiff)].iloc[0, 18]
                rvry = (vp - vm)/(2.0*pivPixelDiff*self.um_pix)
            elif fyp:
                vp = df2[(df2['x'] == x) & (df2['y'] == y + pivPixelDiff)].iloc[0, 18]
                rvry = (vp - v)/(pivPixelDiff*self.um_pix)
            elif fym:
                vm = df2[(df2['x'] == x) & (df2['y'] == y - pivPixelDiff)].iloc[0, 18]
                rvry = (v - vm)/(pivPixelDiff*self.um_pix)
            else:
                rvry = np.nan

            df.loc[index, 'divergence'] = rurx + rvry

        X = np.array(df['x']).reshape(62, 62)
        Y = np.array(df['y']).reshape(62, 62)
        D = np.array(df['divergence']).reshape(62, 62)
        vmin = -1.0e-1
        vmax = +1.0e-1
        levels = np.linspace(vmin, vmax, 51)

        fig = plt.figure(frameon=False)
        plt.imshow(self.imgCell, cmap="gray")
        #plt.scatter(df['x'], df['y'], s=1, c=df['divergence'], norm=Normalize(vmin=-5.0e-3, vmax=5.0e-3))
        c = plt.contourf(X, Y, D, levels=levels, cmap='coolwarm', alpha=.2, extend='both', antialiased=True)
        cbar = fig.colorbar(c, ticks=[vmin, vmin/2.0, 0.0, vmax/2.0, vmax])
        cbar.solids.set(alpha=1)

        plt.axis("off")

        #plt.show()
        fig.savefig(f'result/div{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=208.0)
        plt.close()

    def Defect(self):
        self.defectPixelDiff = 20

        self.dfo = pd.read_csv(f'../../imagej/orientation/data{self.idTarget:04}.csv')
        self.dfo['Orientation'] = self.dfo['Orientation']/180.0*np.pi
        self.dfo['isInsideMask'] = True # 8
        self.dfo['nx'] = 0.0; self.dfo['ny'] = 0.0 # 9, 10

        defect_x = []; defect_y = []; defect_c = []

        # q tensor
        q = np.zeros((2, 2))

        for index, row in self.dfo.iterrows():
            x = int(row['X'])
            y = int(row['Y'])

            if x >= self.imgMask0.shape[1] or y >= self.imgMask0.shape[0]:
                self.dfo.iloc[index, 8] = False
                continue

            if self.imgMask0[y][x] == 0:
                self.dfo.iloc[index, 8] = False
                continue

            theta = row["Orientation"]
            coherency = row["Coherency"]
            tx = np.cos(theta)
            ty = np.sin(theta)
            self.dfo.iloc[index,  9] = coherency*tx
            self.dfo.iloc[index, 10] = coherency*ty

            q[0][0] += tx*tx - 0.5 
            q[0][1] += tx*ty 
            q[1][0] += ty*tx 
            q[1][1] += ty*ty - 0.5 

            if self.isDefectComputable(x, y):# and coherency < 0.5:
                angle = 0.0
                angle += self.angleDiff(x, y, +1, +0, +1, +1)
                angle += self.angleDiff(x, y, +1, +1, +0, +1)
                angle += self.angleDiff(x, y, +0, +1, -1, +1)
                angle += self.angleDiff(x, y, -1, +1, -1, +0)
                angle += self.angleDiff(x, y, -1, +0, -1, -1)
                angle += self.angleDiff(x, y, -1, -1, +0, -1)
                angle += self.angleDiff(x, y, +0, -1, +1, -1)
                angle += self.angleDiff(x, y, +1, -1, +1, +0)
                
                if abs(angle/(2.0*np.pi)) > 1.0e-1:
                    defect_x.append(row["X"])
                    defect_y.append(row["Y"])
                    defect_c.append(-angle/(2.0*np.pi))

        # build new data frame
        df2 = self.dfo[self.dfo['isInsideMask']]
        evalue, evec = np.linalg.eig(2.0*q/len(df2))

        fig = plt.figure(frameon=False)
        plt.imshow(self.imgCell, cmap="gray")
        plt.quiver(df2['X'], df2['Y'], df2['nx'], df2['ny'],
                   color='y', scale_units='xy', pivot='middle',
                   scale=4.0e-2, width=2.5e-3, headaxislength=0, headlength=0)
        plt.scatter(defect_x, defect_y, s=5, c=defect_c, cmap='coolwarm', norm=Normalize(vmin=-0.5, vmax=0.5))
        plt.axis("off")

        #plt.show()
        fig.savefig(f'result/defect{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=277.2)
        plt.close()
        
        return evalue.max()

    def isDefectComputable(self, x, y):
        count = 0

        count += self.isReliableOrientation(x, y, +1, +0)
        count += self.isReliableOrientation(x, y, +1, +1)
        count += self.isReliableOrientation(x, y, +0, +1)
        count += self.isReliableOrientation(x, y, -1, +1)
        count += self.isReliableOrientation(x, y, -1, +0)
        count += self.isReliableOrientation(x, y, -1, -1)
        count += self.isReliableOrientation(x, y, +0, -1)
        count += self.isReliableOrientation(x, y, +1, -1)

        if count == 8:
            return True
        else:
            return False

    def isReliableOrientation(self, x, y, dx, dy):
        tx = x + dx*self.defectPixelDiff
        ty = y + dy*self.defectPixelDiff

        tmp = ((self.dfo['X'] == tx) & (self.dfo['Y'] == ty))
        if(tmp.sum() != 0 and self.dfo[tmp].iloc[0, 6] > 0.10 and self.imgMask0[ty][tx] == 255):
            return True
        else:
            return False

    def angleDiff(self, x, y, dx1, dy1, dx2, dy2):
        tmp1 = ((self.dfo['X'] == x + dx1*self.defectPixelDiff) & (self.dfo['Y'] == y + dy1*self.defectPixelDiff))
        tmp2 = ((self.dfo['X'] == x + dx2*self.defectPixelDiff) & (self.dfo['Y'] == y + dy2*self.defectPixelDiff))

        diff = self.dfo[tmp2].iloc[0, 5] - self.dfo[tmp1].iloc[0, 5]

        if diff > +np.pi/2.0: diff -= np.pi
        if diff < -np.pi/2.0: diff += np.pi

        return diff

    def Q_tensor(self, theta):
        q = np.zeros((2, 2))
        tx = np.cos(theta)
        ty = np.sin(theta)

        q[0][0] += tx*tx - 0.5 
        q[0][1] += tx*ty 
        q[1][0] += ty*tx 
        q[1][1] += ty*ty - 0.5 

        return q 

    def GetNeighbourIndex(self, df, x, y, diff, nx, ny):
        temp = df.index[(df['X'] == x + nx*diff) & (df['Y'] == y + ny*diff)].tolist()

        if not temp:
            return False
        else:
            return temp[0]

    def Defect2(self):
        self.defectPixelDiff = 20

        df = pd.read_csv(f'../../imagej/orientation/data{self.idTarget:04}.csv')
        df['Orientation'] = df['Orientation']/180.0*np.pi
        df['isInsideMask'] = True # 8
        df['nx'] = 0.0; df['ny'] = 0.0 # 9, 10

        defect_x = []; defect_y = []; defect_c = []

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

        df2 = df[df['isInsideMask']]
        for index, row in df2.iterrows():
            x = int(row['X'])
            y = int(row['Y'])

            ixp = self.GetNeighbourIndex(df2, x, y, self.defectPixelDiff, +1, +0)
            ixm = self.GetNeighbourIndex(df2, x, y, self.defectPixelDiff, -1, +0)
            iyp = self.GetNeighbourIndex(df2, x, y, self.defectPixelDiff, +0, +1)
            iym = self.GetNeighbourIndex(df2, x, y, self.defectPixelDiff, +0, -1)

            Q0 = self.Q_tensor(df2.loc[index, 'Orientation'])
            if ixp and ixm:
                Qxp = self.Q_tensor(df2.loc[ixp, 'Orientation'])
                Qxm = self.Q_tensor(df2.loc[ixm, 'Orientation'])
                rQxx_rx = (Qxp[0][0] - Qxm[0][0])/(2.0*self.defectPixelDiff*self.um_pix)
                rQxy_rx = (Qxp[0][1] - Qxm[0][1])/(2.0*self.defectPixelDiff*self.um_pix)
            elif ixp:
                Qxp = self.Q_tensor(df2.loc[ixp, 'Orientation'])
                rQxx_rx = (Qxp[0][0] - Q0[0][0])/(self.defectPixelDiff*self.um_pix)
                rQxy_rx = (Qxp[0][1] - Q0[0][1])/(self.defectPixelDiff*self.um_pix)
            elif ixm:
                Qxm = self.Q_tensor(df2.loc[ixm, 'Orientation'])
                rQxx_rx = (Q0[0][0] - Qxm[0][0])/(self.defectPixelDiff*self.um_pix)
                rQxy_rx = (Q0[0][1] - Qxm[0][1])/(self.defectPixelDiff*self.um_pix)
            else:
                rQxx_rx = np.nan
                rQxy_rx = np.nan

            if iyp and iym:
                Qyp = self.Q_tensor(df2.loc[iyp, 'Orientation'])
                Qym = self.Q_tensor(df2.loc[iym, 'Orientation'])
                rQxx_ry = (Qyp[0][0] - Qym[0][0])/(2.0*self.defectPixelDiff*self.um_pix)
                rQxy_ry = (Qyp[0][1] - Qym[0][1])/(2.0*self.defectPixelDiff*self.um_pix)
            elif iyp:
                Qyp = self.Q_tensor(df2.loc[iyp, 'Orientation'])
                rQxx_ry = (Qyp[0][0] - Q0[0][0])/(self.defectPixelDiff*self.um_pix)
                rQxy_ry = (Qyp[0][1] - Q0[0][1])/(self.defectPixelDiff*self.um_pix)
            elif iym:
                Qym = self.Q_tensor(df2.loc[iym, 'Orientation'])
                rQxx_ry = (Q0[0][0] - Qym[0][0])/(self.defectPixelDiff*self.um_pix)
                rQxy_ry = (Q0[0][1] - Qym[0][1])/(self.defectPixelDiff*self.um_pix)
            else:
                rQxx_ry = np.nan
                rQxy_ry = np.nan

            defect_charge = (rQxx_rx*rQxy_ry - rQxx_ry*rQxy_rx)/(2.0*np.pi)

            if abs(defect_charge) > 1.0e-1:
                defect_x.append(x)
                defect_y.append(y)
                defect_c.append(defect_charge)

        fig = plt.figure(frameon=False)
        plt.imshow(self.imgCell, cmap="gray")
        plt.quiver(df2['X'], df2['Y'], df2['nx'], df2['ny'],
                   color='y', scale_units='xy', pivot='middle',
                   scale=4.0e-2, width=2.5e-3, headaxislength=0, headlength=0)
        plt.scatter(defect_x, defect_y, s=5, c=defect_c, cmap='coolwarm', norm=Normalize(vmin=-0.5, vmax=0.5))
        plt.axis("off")

        #plt.show()
        fig.savefig(f'result/defect{self.idImage:04}.png', bbox_inches='tight', pad_inches=0, dpi=277.2)
        plt.close()
