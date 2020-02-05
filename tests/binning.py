#

# 3x3: SALT/TX_Col/20150401/product/bxgpS201504010002.fits
# 4x4: SALT/2016-2-DDT-006/0209/bxgpS201702090016.fits
# 6x6: /data/sources/J1928-5001/SALT/20140801/product/bxgpS201408010003.fits

from pathlib import Path
from salticam.slotmode.extract import SlotModeExtract, ALL_CHANNELS
from graphing.imagine import plot_image_grid

import matplotlib.pyplot as plt

import numpy as np

root = Path('/media/Oceanus/UCT/Observing/')
filenames = {3: root / 'SALT/TX_Col/20150401/product/',
             4: root / 'SALT/2016-2-DDT-006/0209/',
             6: root / 'data/sources/J1928-5001/SALT/20140801/product/'}

stack = []
for binning, path in filenames.items():
    ex = SlotModeExtract([next(path.glob('*.fits'))])
    n_dud, data, info = ex.detect_noise_frames(ALL_CHANNELS)
    thumbnails = np.median(data[:n_dud, :, :, :data.shape[2]], 0)
    thumbnails -= np.median(thumbnails, (1, 2), keepdims=True)
    thumbnails /= np.std(thumbnails, (1, 2), keepdims=True)
    stack.extend(thumbnails)

fig, axes, imd = plot_image_grid(stack, (len(filenames), 4))

plt.show()