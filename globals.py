COMPONENTS_PER_EFFECT = 8
LIST_OF_EFFECTS = ['Clear Throat', 'Cough', 'Door Slam', 'Drawer', 'Keyboard', 'Keys Drop', 'Knock', 'Laughter', 'Page Turn', 'Phone', 'Speech', 'Other']

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
