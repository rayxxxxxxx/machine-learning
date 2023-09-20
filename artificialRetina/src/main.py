import glob
from pathlib import Path

import numpy as np
from numba import njit

from PIL import Image
from matplotlib import pyplot as plt

from modules.convolution_fade_mem_nn import ConvolutionFadeMemoryNN

import vars


def load_images(path: Path) -> list[np.ndarray]:
    files = glob.glob(path.as_posix())
    images = []
    for f in files:
        img = Image.open(f).convert('L')
        img.thumbnail(vars.MEMSHAPE,
                      Image.Resampling.LANCZOS)
        img = np.array(img).transpose().astype('float64')
        images.append(img/np.max(img))
    return images


def recunstruct_mem(cfmnn: ConvolutionFadeMemoryNN):
    r = np.zeros(cfmnn.S.shape)
    for i in range(cfmnn.S.shape[0]-1):
        for j in range(cfmnn.S.shape[1]-1):
            r[i:i+2, j:j+2] += cfmnn.W[i][j]
    return r.transpose()


def main():
    cfmnn = ConvolutionFadeMemoryNN(
        vars.MEMSHAPE, vars.P, vars.ALPHA, vars.BETA)

    # images = load_images(Path('src', 'images', 'pictures', '*'))
    images = load_images(Path('src', 'images', 'numbers', '*'))

    for k in range(1):
        for i in range(10):
            cfmnn.update(images[i] + np.random.uniform(-vars.NOISE,
                                                       vars.NOISE, vars.MEMSHAPE))

    fig1, axes1 = plt.subplots(1, figsize=(10, 7), dpi=100)
    fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(10, 7), dpi=100)

    axes1.matshow(recunstruct_mem(cfmnn), cmap='inferno')

    for i in range(3):
        for j in range(3):
            index = 3*i+j
            if index < len(images):
                img = images[index] + \
                    np.random.uniform(-vars.NOISE, vars.NOISE, vars.MEMSHAPE)
                res = cfmnn.predict(img).transpose()
                axes2[i][j].matshow(res/np.max(res), cmap='inferno')

    plt.show()


if __name__ == '__main__':
    main()
