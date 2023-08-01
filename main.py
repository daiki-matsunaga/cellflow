import sys
import tqdm
import configparser
from module.celldata import CellData

class Config:
    def __init__(self, argv):
        if len(argv) != 2:
            sys.exit('Error: Pass INI file to the command line argument')

        runcard = configparser.ConfigParser()
        runcard.read(argv[1])

        # read parameters from runcard
        self.DIR = runcard['global']['DIR']
        self.UM_PIX = 1.0/float(runcard['global']['PIX_UM'])
        self.PIV_FRAME_DIFF = runcard.getint('piv', 'PIV_FRAME_DIFF')
        self.FRAME_INTERVAL = runcard.getfloat('piv', 'FRAME_INTERVAL')

        self.TARGET_U = {'x': 'ux1', 'y': 'uy1', 'mag': 'mag1'}
        if runcard['piv']['TARGET_U'] == 'ux0':
            self.TARGET_U = {'x': 'ux0', 'y': 'uy0', 'mag': 'mag0'}

        # read options
        self.FLAG_SUBTRACT_AVERAGE_PIV      = runcard.getboolean('options', 'FLAG_SUBTRACT_AVERAGE_PIV')      
        self.FLAG_COMPUTE_VELOCITY_FIELD    = runcard.getboolean('options', 'FLAG_COMPUTE_VELOCITY_FIELD')    
        self.FLAG_COMPUTE_ORIENTATION_FIELD = runcard.getboolean('options', 'FLAG_COMPUTE_ORIENTATION_FIELD') 

if __name__ == '__main__':
    # read configurations from runcard
    config = Config(sys.argv)
    data = CellData(config)

    with tqdm.tqdm(range(900)) as pbar:
#    with tqdm.tqdm(range(1)) as pbar:
#    with tqdm.tqdm(range(50)) as pbar:
        for idImage in pbar:
            pbar.set_description(f'[image number {idImage:04}]')

            data.compute(idImage)

#    data.draw_figure(data.intensity_mean, 'mean intensity')
#    data.draw_figure(data.average_velocity, 'average velocity')
