import sys
import tqdm
import yaml
from module.celldata import CellData

import pickle

def read_runcard(argv):
    if len(argv) != 2:
        sys.exit('Error: Pass json configuration file to the command line argument')

    with open(argv[1]) as f:
        config = yaml.safe_load(f)

    config['general']['UM_PIX'] = 1.0/config['general']['PIX_UM']
    return config

if __name__ == '__main__':
    # read configurations from runcard
    config = read_runcard(sys.argv)
    data = CellData(config)

    with tqdm.tqdm(range(900)) as pbar:
#    with tqdm.tqdm(range(1)) as pbar:
#    with tqdm.tqdm(range(50)) as pbar:
        for idImage in pbar:
            pbar.set_description(f'[image number {idImage:04}]')

            data.compute(idImage)

#    data.draw_figure(data.intensity_mean, 'mean intensity')
#    data.draw_figure(data.average_velocity, 'average velocity')
