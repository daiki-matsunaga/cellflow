%%time
import pandas as pd
import glob
import os
from pathlib import Path
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage import io
import cv2
from PIL import Image

interval = 30

path = "E:\\file\\dm_data"
oj_path = path + "\\orientation\\vector_csv"
os.makedirs(oj_path, exist_ok=True)
csv_files = glob.glob(oj_path + "\\bandpass8\\*.csv")
path_crop = path + "\\crop"
img_path = path + "\\scale_png"
img_files = glob.glob(img_path + "\\scale_*.png")
img_files.sort()
path_nucle = path + "\\nucleation_tif"

def queue_defect(df): # add queue in defect rows for calculating average positions
    df['queue_plus'] = False
    df['queue_minus'] = False
    df.loc[(df['defect'] < -0.4), 'queue_minus'] = True
    df.loc[(0.4 < df['defect']), 'queue_plus'] = True
    return df

def average_position(df, intvl): # average defect positions
    x_ave = []
    y_ave = []
    c = [] # set list to append defect position for scatter plot
    
    # plus ver
    df_plus = df.loc[df['queue_plus'] == True]
    df_plus.reset_index(inplace=True, drop=True)
    for i in range(len(df_plus)):
        if df_plus['queue_plus'][i] == True:
            x_temp = []
            y_temp = []
            x, y = df_plus.at[i, 'X'], df_plus.at[i, 'Y']
            x_temp.append(x)
            y_temp.append(y)
            df_xm = df[(df['X'] == x - intvl) & (df['Y'] == y)]
            df_ym = df[(df['X'] == x) & (df['Y'] == y - intvl)]
            df_xmym = df[(df['X'] == x - intvl) & (df['Y'] == y - intvl)]
            df_xp = df[(df['X'] == x + intvl) & (df['Y'] == y)]
            df_yp = df[(df['X'] == x) & (df['Y'] == y + intvl)]
            df_xpyp = df[(df['X'] == x + intvl) & (df['Y'] == y + intvl)]
            df_xpym = df[(df['X'] == x + intvl) & (df['Y'] == y - intvl)]
            df_xmyp = df[(df['X'] == x - intvl) & (df['Y'] == y + intvl)]

            df_xm.reset_index(inplace=True, drop=True)
            df_ym.reset_index(inplace=True, drop=True)
            df_xmym.reset_index(inplace=True, drop=True)
            df_xp.reset_index(inplace=True, drop=True)
            df_yp.reset_index(inplace=True, drop=True)
            df_xpyp.reset_index(inplace=True, drop=True)
            df_xpym.reset_index(inplace=True, drop=True)
            df_xmyp.reset_index(inplace=True, drop=True)
            
            
            # switch queue from True to False
            if df_xm.at[0, 'queue_plus'] == True:
                x_temp.append(x - intvl)
                y_temp.append(y)
                df.queue_plus[(df.X == x - intvl) & (df.Y == y)] = False
                df_plus.queue_plus[(df_plus.X == x - intvl) & (df_plus.Y == y)] = False
            if df_ym.at[0, 'queue_plus'] == True:
                x_temp.append(x)
                y_temp.append(y - intvl)
                df.queue_plus[(df.X == x) & (df.Y == y - intvl)] = False
                df_plus.queue_plus[(df_plus.X == x) & (df_plus.Y == y - intvl)] = False
            if df_xmym.at[0, 'queue_plus'] == True:
                x_temp.append(x - intvl)
                y_temp.append(y - intvl)
                df.queue_plus[(df.X == x - intvl) & (df.Y == y - intvl)] = False
                df_plus.queue_plus[(df_plus.X == x - intvl) & (df_plus.Y == y - intvl)] = False
            if df_xp.at[0, 'queue_plus'] == True:
                x_temp.append(x + intvl)
                y_temp.append(y)
                df.queue_plus[(df.X == x + intvl) & (df.Y == y)] = False
                df_plus.queue_plus[(df_plus.X == x + intvl) & (df_plus.Y == y)] = False
            if df_yp.at[0, 'queue_plus'] == True:
                x_temp.append(x)
                y_temp.append(y + intvl)
                df.queue_plus[(df.X == x) & (df.Y == y + intvl)] = False
                df_plus.queue_plus[(df_plus.X == x) & (df_plus.Y == y + intvl)] = False
            if df_xpyp.at[0, 'queue_plus'] == True:
                x_temp.append(x + intvl)
                y_temp.append(y + intvl)
                df.queue_plus[(df.X == x + intvl) & (df.Y == y + intvl)] = False
                df_plus.queue_plus[(df_plus.X == x + intvl) & (df_plus.Y == y + intvl)] = False
            if df_xpym.at[0, 'queue_plus'] == True:
                x_temp.append(x + intvl)
                y_temp.append(y - intvl)
                df.queue_plus[(df.X == x + intvl) & (df.Y == y - intvl)] = False
            if df_xmyp.at[0, 'queue_plus'] == True:
                x_temp.append(x - intvl)
                y_temp.append(y + intvl)
                df.queue_plus[(df.X == x - intvl) & (df.Y == y + intvl)] = False
                df_plus.queue_plus[(df_plus.X == x - intvl) & (df_plus.Y == y + intvl)] = False
            # append average posision
            x_ave.append((sum(x_temp)) / (len(x_temp)))
            y_ave.append(sum(y_temp) / len(y_temp))
            c.append(0.5)
            df_plus.iloc[i, 9] = False
            df.queue_plus[(df.X == x) & (df.Y == y)] = False
    
    
    # minus ver
    df_minus = df.loc[df['queue_minus'] == True]
    df_minus.reset_index(inplace=True, drop=True)
    for i in range(len(df_minus)):
        if df_minus.iloc[i, 10] == True:
            x_temp = []
            y_temp = []
            x, y = df_minus.iloc[i, 0], df_minus.iloc[i, 1]
            x_temp.append(x)
            y_temp.append(y)
            df_xm = df[(df['X'] == x - intvl) & (df['Y'] == y)]
            df_ym = df[(df['X'] == x) & (df['Y'] == y - intvl)]
            df_xmym = df[(df['X'] == x - intvl) & (df['Y'] == y - intvl)]
            df_xp = df[(df['X'] == x + intvl) & (df['Y'] == y)]
            df_yp = df[(df['X'] == x) & (df['Y'] == y + intvl)]
            df_xpyp = df[(df['X'] == x + intvl) & (df['Y'] == y + intvl)]
            df_xpym = df[(df['X'] == x + intvl) & (df['Y'] == y - intvl)]
            df_xmyp = df[(df['X'] == x - intvl) & (df['Y'] == y + intvl)]

            df_xm.reset_index(inplace=True, drop=True)
            df_ym.reset_index(inplace=True, drop=True)
            df_xmym.reset_index(inplace=True, drop=True)
            df_xp.reset_index(inplace=True, drop=True)
            df_yp.reset_index(inplace=True, drop=True)
            df_xpyp.reset_index(inplace=True, drop=True)
            df_xpym.reset_index(inplace=True, drop=True)
            df_xmyp.reset_index(inplace=True, drop=True)
            
            
            # switch queue from True to False
            if df_xm.at[0, 'queue_minus'] == True:
                x_temp.append(x - intvl)
                y_temp.append(y)
                df.queue_minus[(df.X == x - intvl) & (df.Y == y)] = False
                df_minus.queue_minus[(df_minus.X == x - intvl) & (df_minus.Y == y)] = False
            if df_ym.at[0, 'queue_minus'] == True:
                x_temp.append(x)
                y_temp.append(y - intvl)
                df.queue_minus[(df.X == x) & (df.Y == y - intvl)] = False
                df_minus.queue_minus[(df_minus.X == x) & (df_minus.Y == y - intvl)] = False
            if df_xmym.at[0, 'queue_minus'] == True:
                x_temp.append(x - intvl)
                y_temp.append(y - intvl)
                df.queue_minus[(df.X == x - intvl) & (df.Y == y - intvl)] = False
                df_minus.queue_minus[(df_minus.X == x - intvl) & (df_minus.Y == y - intvl)] = False
            if df_xp.at[0, 'queue_minus'] == True:
                x_temp.append(x + intvl)
                y_temp.append(y)
                df.queue_minus[(df.X == x + intvl) & (df.Y == y)] = False
                df_minus.queue_minus[(df_minus.X == x + intvl) & (df_minus.Y == y)] = False
            if df_yp.at[0, 'queue_minus'] == True:
                x_temp.append(x)
                y_temp.append(y + intvl)
                df.queue_minus[(df.X == x) & (df.Y == y + intvl)] = False
                df_minus.queue_minus[(df_minus.X == x) & (df_minus.Y == y + intvl)] = False
            if df_xpyp.at[0, 'queue_minus'] == True:
                x_temp.append(x + intvl)
                y_temp.append(y + intvl)
                df.queue_minus[(df.X == x + intvl) & (df.Y == y + intvl)] = False
                df_minus.queue_minus[(df_minus.X == x + intvl) & (df_minus.Y == y + intvl)] = False
            if df_xpym.at[0, 'queue_minus'] == True:
                x_temp.append(x + intvl)
                y_temp.append(y - intvl)
                df.queue_minus[(df.X == x + intvl) & (df.Y == y - intvl)] = False
            if df_xmyp.at[0, 'queue_minus'] == True:
                x_temp.append(x - intvl)
                y_temp.append(y + intvl)
                df.queue_minus[(df.X == x - intvl) & (df.Y == y + intvl)] = False
                df_minus.queue_minus[(df_minus.X == x - intvl) & (df_minus.Y == y + intvl)] = False
            # append average posision
            x_ave.append((sum(x_temp)) / (len(x_temp)))
            y_ave.append(sum(y_temp) / len(y_temp))
            c.append(-0.5)
            df_minus.iloc[i, 10] = False
            df.queue_minus[(df.X == x) & (df.Y == y)] = False
    
    return(x_ave, y_ave, c)


for f in range(len(csv_files)):
    print(f)
    df = pd.read_csv(csv_files[f])
    num_f = str(f).zfill(4)
    
    crop_img = Image.open(path_crop + '\\image'+ num_f + '.tif')
    crop_img_array = np.array(crop_img)
    df["defect"] = 0
    for i in range(len(df)):
            x_point = df.at[i, 'X']
            y_point = df.at[i, 'Y']
            if x_point > 1024 or y_point > 1024:
                df = df.drop([i])
                continue
            elif crop_img_array[y_point, x_point] == 0:
                df = df.drop([i])


    for num in range(len(df)):
        # pick up orientation data in surrounding area

        orientation = []
        df_add_x = df[df['X'] == df.iloc[num, 0] + interval]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1]]
        if df_add_xy.size == 0:
            continue
        orientation.append(df_add_xy.iloc[0, 5])

        df_add_x = df[df['X'] == df.iloc[num, 0] + interval]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1] - interval]
        if df_add_xy.size == 0:
            continue
        orientation.append(df_add_xy.iloc[0, 5])

        df_add_x = df[df['X'] == df.iloc[num, 0]]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1] - interval]
        if df_add_xy.size == 0:
            continue
        orientation.append(df_add_xy.iloc[0, 5])

        df_add_x = df[df['X'] == df.iloc[num, 0] - interval]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1] - interval]
        if df_add_xy.size == 0:
            continue    
        orientation.append(df_add_xy.iloc[0, 5])

        df_add_x = df[df['X'] == df.iloc[num, 0] - interval]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1]]
        if df_add_xy.size == 0:
            continue
        orientation.append(df_add_xy.iloc[0, 5])


        df_add_x = df[df['X'] == df.iloc[num, 0] - interval]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1] + interval]
        if df_add_xy.size == 0:
            continue    
        orientation.append(df_add_xy.iloc[0, 5])

        df_add_x = df[df['X'] == df.iloc[num, 0]]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1] + interval]
        if df_add_xy.size == 0:
            continue
        orientation.append(df_add_xy.iloc[0, 5])

        df_add_x = df[df['X'] == df.iloc[num, 0] + interval]
        df_add_xy = df_add_x[df_add_x['Y'] == df.iloc[num, 1] + interval]
        if df_add_xy.size == 0:
            continue
        orientation.append(df_add_xy.iloc[0, 5])

        # calcurate defect
        d = 0
        for i in range(len(orientation)):
            if i == len(orientation)-1:
                d = (orientation[i] - orientation[0])
            else:
                d = (orientation[i] - orientation[i+1])
            if d < -90:
                d += 1/2
            elif d > 90:
                d -= 1/2
            df.iloc[num, 8] -= d
    
        # resize defect data into xy np array
    size_x = 35 # (1037 - 17)/30 + 1
    size_y = 35
    defect_img = np.zeros((size_x, size_y))
    interval = 30 # pixel
    initial_pixel = 17
    for j in range(len(df)):
        x_int = int((df.iloc[j, 1] - initial_pixel) / interval)
        y_int = int((df.iloc[j, 0] - initial_pixel) / interval)
        defect_img[x_int, y_int] = df.iloc[j, 8]

    
    # show defect points 
    cm = plt.cm.get_cmap('Greens')
    img = plt.imread(img_files[f])
    plt.imshow(img, cmap = "Greens")
    
    df.reset_index(inplace=True, drop=True)
    df = queue_defect(df)
    
    df.reset_index(inplace=True, drop=True)
    average = average_position(df, interval)
    
    # visualize orientation
    ori_x = df['X'].to_numpy()
    ori_y = df['Y'].to_numpy()
    dx = df['DX'].to_numpy()
    dy = df['DY'].to_numpy()
    angle = df['Orientation'].to_numpy()
    plt.quiver(ori_x, ori_y, dx, -dy, pivot='mid', color='y', width=0.003, headwidth=0, headaxislength=0, headlength=0, scale=40)
    
    pil_image = Image.fromarray(defect_img)
    defect_img_resize = np.asarray(pil_image.resize((img.shape[0], img.shape[1])))
    defect_img_resize = np.where((defect_img_resize > -0.4) & (0.4 > defect_img_resize)  , np.nan, defect_img_resize)
    plt.axis("off")
    # plot nucleation position
    
    df_defect = df[(df['defect'] > 0.4) | (df['defect'] < -0.4)]
    plt.scatter(average[0], average[1], c = average[2], cmap = 'bwr',s = 5, vmax=0.5, vmin=-0.5, alpha = 0.5)
    plt.savefig(path + "\\orientation\\defect_img\\bp8\\ave8_defect_" + num_f + "scale.png", bbox_inches='tight', pad_inches=0, dpi=600)
    print('finish:',num_f)
    plt.show()
