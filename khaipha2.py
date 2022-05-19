import matplotlib

# matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image

# import PIL
# from PIL import image
# %matplotlib inline
plt.style.use("ggplot")

from skimage import io
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_sample_image
import seaborn as sns;

sns.set()
import os

for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Note: this requires the ``pillow`` package to be installed
dog = imread('../input/film-image/dog.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(dog);
plt.show();
print(dog.shape)
print(dog.size)

fox = imread('../input/film-image/fox.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(fox);
plt.show();
print(fox.shape)
print(fox.size)

lion = imread('../input/film-image/lion.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(lion);
plt.show();
print(lion.shape)
print(lion.size)

tiger = imread('../input/film-image/tiger2.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(tiger);
plt.show();
print(tiger.shape)
print(tiger.size)

wolf = imread('../input/film-image/wolf.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(wolf);
plt.show();
print(wolf.shape)
print(wolf.size)

panda = imread('../input/film-image/Tuan.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(panda);
plt.show();
print(panda.shape)
print(panda.size)

doggy = dog / 255.0  # use 0...1 scale
doggy = doggy.reshape(381 * 271, 3)
doggy.shape

foxy = fox / 255.0  # use 0...1 scale
foxy = foxy.reshape(298 * 203, 3)
foxy.shape

liony = lion / 255.0  # use 0...1 scale
liony = liony.reshape(275 * 220, 3)
liony.shape

tigery = tiger / 255.0  # use 0...1 scale
tigery = tigery.reshape(286 * 304, 3)
tigery.shape

wolfy = wolf / 255.0  # use 0...1 scale
wolfy = wolfy.reshape(328 * 302, 3)
wolfy.shape

panday = panda / 255.0  # use 0...1 scale
panday = panday.reshape(1920 * 1080, 3)
panday.shape


def plot_pixels(doggy, title, colors=None, N=10000):
    if colors is None:
        colors = doggy

    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(doggy.shape[0])[:N]
    colors = colors[i]
    R, G, B = doggy[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);


plot_pixels(doggy, title='Input color space: 16 million possible colors')
plt.show();
import warnings;

warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(doggy)
new_colors = kmeans.cluster_centers_[kmeans.predict(doggy)]

plot_pixels(doggy, colors=new_colors,
            title="Reduced color space: 16 colors")
plt.show();
dog_recolored = new_colors.reshape(dog.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(dog)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(dog_recolored)
ax[1].set_title('16-color Image', size=16);

# store to file
plt.savefig("dog_kmean.png", dpi=125)
plt.show();

plot_pixels(foxy, title='Input color space: 16 million possible colors')
plt.show();
import warnings;

warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(foxy)
new_colors = kmeans.cluster_centers_[kmeans.predict(foxy)]

plot_pixels(foxy, colors=new_colors,
            title="Reduced color space: 16 colors")
plt.show();

fox_recolored = new_colors.reshape(fox.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(fox)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(fox_recolored)
ax[1].set_title('16-color Image', size=16);

# store to file
plt.savefig("fox_kmean.png", dpi=125)
plt.show();

plot_pixels(liony, title='Input color space: 16 million possible colors')
plt.show();
import warnings;

warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(liony)
new_colors = kmeans.cluster_centers_[kmeans.predict(liony)]

plot_pixels(liony, colors=new_colors,
            title="Reduced color space: 16 colors")
plt.show();

lion_recolored = new_colors.reshape(lion.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(lion)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(lion_recolored)
ax[1].set_title('16-color Image', size=16);

# store to file
plt.savefig("lion_kmean.png", dpi=125)
plt.show();

plot_pixels(tigery, title='Input color space: 16 million possible colors')
plt.show();
import warnings;

warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(tigery)
new_colors = kmeans.cluster_centers_[kmeans.predict(tigery)]

plot_pixels(tigery, colors=new_colors,
            title="Reduced color space: 16 colors")
plt.show();

tiger_recolored = new_colors.reshape(tiger.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(tiger)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(tiger_recolored)
ax[1].set_title('16-color Image', size=16);

# store to file
plt.savefig("tiger_kmean.png", dpi=125)
plt.show();

plot_pixels(wolfy, title='Input color space: 16 million possible colors')
plt.show();
import warnings;

warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(wolfy)
new_colors = kmeans.cluster_centers_[kmeans.predict(wolfy)]

plot_pixels(wolfy, colors=new_colors,
            title="Reduced color space: 16 colors")
plt.show();
wolf_recolored = new_colors.reshape(wolf.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(wolf)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(wolf_recolored)
ax[1].set_title('16-color Image', size=16);

# store to file
plt.savefig("wolf_kmean.png", dpi=125)
plt.show();

plot_pixels(panday, title='Input color space: 16 million possible colors')
plt.show();
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(panday)
new_colors = kmeans.cluster_centers_[kmeans.predict(panday)]

plot_pixels(panday, colors=new_colors,
            title="Reduced color space: 16 colors")
plt.show();
panda_recolored = new_colors.reshape(panda.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(panda)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(panda_recolored)
ax[1].set_title('16-color Image', size=16);

# store to file
plt.savefig("Tuan_kmean.png", dpi=125)
plt.show();
