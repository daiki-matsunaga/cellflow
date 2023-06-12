import pandas as pd

def apply_mask(df, imgMask): 
    # extract inner part (set 'isInsideMask' to False if outside)
    for index, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])

        if imgMask[y][x] == 0:
            df.loc[index, 'isInsideMask'] = False

    return df[df['isInsideMask']]
