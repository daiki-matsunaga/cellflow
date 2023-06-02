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


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def make_contour(file, threshold):
    img=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    thresh, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    # contours:positions of outlines
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # remove small contours by filtering
    contours = list(filter(lambda x: cv2.contourArea(x) > threshold, contours))
    return contours, img
    
def scale_contour(cnt, scale):
    # Getting the centrouid of the
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # translating the contour by subtracting the center with all the points
    cnt_norm = cnt - [cx, cy]

    # scale each point of the contour
    cnt_scaled = cnt_norm * scale

    # translate back the contour to it's original place
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled

def contour_to_mask(img, contours):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contours], -1, color=255, thickness=-1)
    return mask



path = "E:\\file\\20220920\\1min\\1min_Cycle_02"
# path_image = path + "\\gfp_crop"
path_vector = path + "\\piv"
save_path = path_vector + "\\div"#フォルダ名
if not os.path.exists(save_path):#ディレクトリがなかったら
    os.mkdir(save_path)#作成したいフォルダ名を作成
path_fig = path + "\\raw_png"
raw_png_arr = glob.glob(path_fig + "\\*.png")
path_crop_jpg = path + "\\crop_jpg"
mask_files = glob.glob(path_crop_jpg + "\\*.jpg")
mask_files.sort()

files = os.listdir(path_vector)
files_vector = glob.glob(path_vector + "\\*.txt")#  get only files which contain "n_0" and .txt
df_pre = pd.read_csv(files_vector[0], sep=" ", header=None)
# max_min = df_pre[0] -df_pre.min()[0]
max_min = df_pre.iloc[-1,0] - df_pre.iloc[0,0]
dif = df_pre.iloc[1,0] - df_pre.iloc[0,0]
axis = int(max_min / dif + 1)

# set initial number
i = 0
a = 0.5 #alpha 
lim = 0.02 # visualizing limit number

delta_t = 1 # time interval
delta_x = axis
delta_y = axis
div_size = int((1000 - 24) / 16 + 1)
# print(div_size, axis)
pix_um = 4.8272
num = 0

for f in files_vector:
    if num % 50 == 0:
        print(f)
    #calculate div
#     u = (f, sep=" ", header=None)
    u = np.loadtxt(f)
    div = np.zeros((u.shape[0] , 3, ))
    x_int = axis
    y_int = axis
    div_pic = np.zeros((div_size, div_size))
#     print("div, u.size", div, u.size)

    # creat mask
    contours, img = make_contour(mask_files[num], 100)
    cnt_scaled = scale_contour(contours[0], 0.9)
    mask = contour_to_mask(img, cnt_scaled)
    for i in range(u.shape[0]):
        div[i, 0:2] = u[i, 0:2] / pix_um
        
        x_axis = u[i, 0]
        y_axis = u[i, 1]
        
        if i-y_int <= 0 or i+y_int >= u.shape[0]:
            continue
        elif mask[int(y_axis - 1), int(x_axis - 1)] == 0:
            continue
        div_x = (u[i + 1, 2] - u[i - 1, 2]) / (2 * delta_x)
        div_y = (u[i + y_int, 3] - u[i - y_int, 3]) / (2 * delta_y)
        div_pic[i // div_size, i % div_size] = div_x + div_y
        div[i, 2] = div_x + div_y

    zero_i1 = "{0:04d}".format(num)
    zero_i2 = "{0:04d}".format(num+1)
    
    filename_png = save_path + "\\div_png_crop\\" + zero_i1 + "_" + zero_i2 + ".png"
    if not os.path.exists(save_path + "\\div_png_crop"):#ディレクトリがなかったら
        os.mkdir(save_path + "\\div_png_crop")#作成したいフォルダ名を作成
    filename_txt = save_path + "\\div_txt_crop\\" + zero_i1 + "_" + zero_i2 + ".txt"
    if not os.path.exists(save_path + "\\div_txt_crop"):#ディレクトリがなかったら
        os.mkdir(save_path + "\\div_txt_crop")#作成したいフォルダ名を作成
    
    cm = plt.cm.get_cmap('Greens')
    img = plt.imread(raw_png_arr[num])
    plt.imshow(img, cmap = "Greens")
    div_pic = np.where(div_pic == 0, np.nan, div_pic)
    pil_image = Image.fromarray(div_pic)
    div_pic_resize = np.asarray(pil_image.resize((img.shape[1], img.shape[0])))
    np.savetxt(filename_txt, div)
    plt.imshow(div_pic_resize, cmap = 'jet', interpolation = 'none', alpha = a)
    plt.imshow(div_pic_resize, cmap = 'jet', vmin = -lim, vmax = lim,  interpolation = 'none', alpha = a)
    
    plt.colorbar(label="Divegence[-/min]")
    plt.axis("off")
    plt.savefig(filename_png, bbox_inches='tight', pad_inches=0, dpi=243.6)
#     plt.show()
    plt.clf()
    num += 1
    