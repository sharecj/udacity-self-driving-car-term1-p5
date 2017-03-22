# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def plot(image, title, cmap=''):
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.show()
