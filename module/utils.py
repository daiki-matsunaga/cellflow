import pandas as pd

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

def set_isInsideMask(df, imgMask): 
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
