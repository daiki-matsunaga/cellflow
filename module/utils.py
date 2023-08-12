import pandas as pd
import numpy as np

def apply_mask(df, imgMask): 
    df['isInsideMask'] = True
    lx, ly = imgMask.shape

    # extract inner part (set 'isInsideMask' to False if outside)
    for index, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])

        if x >= lx or y >= ly:
            df.loc[index, 'isInsideMask'] = False
            continue

        if imgMask[y][x] == 0:
            df.loc[index, 'isInsideMask'] = False

    return df[df['isInsideMask']]

def is_inside_mask(x, y, imgMask): 
    lx, ly = imgMask.shape

    if (x >= lx or y >= ly) or imgMask[y][x] == 0:
        return False
    else:
        return True

def get_index_at_position(df, x, y):
    tmp = df.index[((df['x'] == x) & (df['y'] == y))].tolist()

    if len(tmp) == 0:
        return False
    else:
        return tmp[0]

def coarse_grain_image(img, numBox):
    imgCoarse = np.empty((numBox, numBox))
    imageSize_y, imageSize_x = img.shape
    boxSize = imageSize_x//numBox

    for bx in range(numBox):
        for by in range(numBox):
            imgCoarse[bx, by] = np.mean(img[bx*boxSize:(bx + 1)*boxSize, by*boxSize:(by + 1)*boxSize])

    return imgCoarse
