import matplotlib.pyplot as plt
from module import analysis
import numpy as np

if __name__ == '__main__':
    v_mean = []
    v_mag = []
    order_param = []
    order_param_v = []
    divergence = []

    for idImage in range(900):
        print(idImage)
        data = analysis.Data(idImage)

        #order_param.append(data.Defect())

        t1, t2, t3 = data.Piv(v_mean)
        v_mag += t1
        divergence += t3
        order_param_v.append(t2)

    '''
    n, bins, patches = plt.hist(divergence, bins=64, range=(-0.02, 0.02), density=1)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(v_mean)
    ax.set(xlabel='frame', ylabel='velocity [um/min]')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(order_param_v)
    ax.set(xlabel='frame', ylabel='order parameter')
    plt.show()

    np.savetxt('order_param.npy', order_param) 
    np.savetxt('order_param_v.npy', order_param_v) 
    np.savetxt('divergence.npy', divergence) 
    np.savetxt('v_mag.npy', v_mag) 
    '''
