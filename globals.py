COMPONENTS_PER_EFFECT = 8
LIST_OF_EFFECTS = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keys', 'knock', 'laughter', 'pageturn', 'phone', 'speech', 'other']

HOP_LENGTH = 256
WINDOW_LENGTH = 512


import matplotlib.pyplot as plt


def colormap(X):
    """
    Displays np array X on a colormap
    """
    fig = plt.figure(figsize=(8, 20))
    plt.pcolor(X)
    plt.colorbar()
    # plt.axes().set_aspect('equal', 'datalim')
    plt.show()
