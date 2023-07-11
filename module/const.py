# *** Directory ***
DIR = '/home/matsunaga/Documents/cellflow/data/samples/220908_0005'

# *** PIV conditions ***
PIV_FRAME_DIFF = 4

# interval between frames [s]
FRAME_INTERVAL = 45.0

UM_PIX = 1.0/4.8272

#TARGET_U = {'x': 'ux0', 'y': 'uy0', 'mag': 'mag0'}
TARGET_U = {'x': 'ux1', 'y': 'uy1', 'mag': 'mag1'}

FLAG_SUBTRACT_AVERAGE_PIV = True
