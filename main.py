import tqdm
from module.celldata import CellData

if __name__ == '__main__':
    data = CellData()

    for idImage in tqdm.tqdm(range(1000)):
        data.read(idImage)

    data.draw_figure(data.intensity_mean, 'mean intensity')
