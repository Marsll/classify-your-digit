import os

import numpy as np
from PIL import Image, ImageOps


def img_to_npy(path_to_file):
    size = (28, 28)
    image = Image.open(path_to_file).convert('L')
    image = ImageOps.invert(image)
    image.thumbnail(size, Image.ANTIALIAS)

    temp_arr = np.array(image)
    idx_argmin = np.unravel_index(temp_arr.argmin(), temp_arr.shape)
    print(idx_argmin)

    color = image.getpixel((idx_argmin[0].item(), idx_argmin[1].item()))

    background = Image.new('L', size, color)
    background.paste(
        image,
        (int((size[0] - image.size[0]) / 2),
         int((size[1] - image.size[1]) / 2)))

    nparr = np.array(background)

    nparr = nparr - nparr.min()
    nparr = nparr/nparr.max()

    out = np.array(nparr, dtype=np.float32)

    filename = os.path.splitext(path_to_file)[0]
    np.save(filename, out)
