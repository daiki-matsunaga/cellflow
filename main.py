import tqdm
from module.celldata import CellData

if __name__ == '__main__':
    data = CellData()

#    with tqdm.tqdm(range(900)) as pbar:
    with tqdm.tqdm(range(1)) as pbar:
        for idImage in pbar:
            pbar.set_description(f'[image number {idImage:04}]')

            data.read(idImage)
            data.output()

#    data.draw_figure(data.intensity_mean, 'mean intensity')
#    data.draw_figure(data.average_velocity, 'average velocity')
