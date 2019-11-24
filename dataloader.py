import numpy as np


def random_crop(img, crop_size):
    H, W = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = np.random.randint(0, W - dx + 1)
    y = np.random.randint(0, H - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop):
    while True:
        batch = next(batches)
        batch_crops = np.zeros((batch.shape[0], crop, crop, 3))
        for i in range(batch.shape[0]):
            batch_crops[i] = random_crop(batch[i], (crop, crop))
        yield batch_crops