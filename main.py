import tqdm
from module.celldata import CellData

if __name__ == '__main__':
    data = CellData()

    with tqdm.tqdm(range(900)) as pbar:
        for idImage in pbar:
            pbar.set_description(f'[image number {idImage:04}]')

            data.read(idImage)

#    data.draw_figure(data.intensity_mean, 'mean intensity')
