import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from numba import cuda
from pylab import cm as cm

global N_pixel, x1_coo, y1_coo, N_coo, L, M, power
L = 800
N_pixel = 1024
ratio_image = 1.
x1_coo = -1.5
y1_coo = -1.5
N_coo = 3.0
i_cmap = 49
power = 2
fig = plt.figure(figsize=(12, 12))
fig.suptitle("Interactive Mandelbrot Set Accelerated using Numba")
ax = fig.add_subplot(111)
cmaps = [m for m in cm.datad if not m.endswith("_r")]

blockdim = (32, 8)
griddim = (32, 16)


@numba.jit
def mandel(x, y):
    c = complex(x, y)
    z = 0.0j
    for i in range(L):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return L


mandel_gpu = cuda.jit(device=True)(mandel)


@cuda.jit
def mandel_kernel(x_0, y_0, N_coo, d_image):
    pixel_size_x = N_coo / N_pixel
    pixel_size_y = N_coo / N_pixel

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, N_pixel, gridX):
        real = x_0 + x * pixel_size_x
        for y in range(startY, N_pixel, gridY):
            imag = y_0 + y * pixel_size_y
            d_image[y, x] = mandel_gpu(real, imag)


def zoom_on_square(eclick, erelease):
    global N_pixel, N_coo, x1_coo, y1_coo, myobj, M, power

    x1_pixel, y1_pixel = min(eclick.xdata, erelease.xdata), min(
        eclick.ydata, erelease.ydata)
    x2_pixel, y2_pixel = max(eclick.xdata, erelease.xdata), max(
        eclick.ydata, erelease.ydata)

    # Get correct ratio zoomed image
    H = (y2_pixel - y1_pixel)
    W = (x2_pixel - x1_pixel)
    ratio_h_w = H / W
    if ratio_h_w > ratio_image:
        difference = (H - W) / 2
        y1_pixel = y1_pixel + difference
        y2_pixel = y2_pixel - difference
    elif ratio_h_w <= ratio_image:
        difference = (W - H) / 2
        x1_pixel = x1_pixel + difference
        x2_pixel = x2_pixel - difference

    assert round(y2_pixel - y1_pixel, 2) == round(x2_pixel - x1_pixel, 2)

    zoom_N_pixel = y2_pixel - y1_pixel
    # Convert from pixel to coordinates in the plane
    ratio_coo_pixel = N_coo / N_pixel
    x1_coo = x1_coo + x1_pixel * ratio_coo_pixel
    y1_coo = y1_coo + y1_pixel * ratio_coo_pixel
    N_coo = zoom_N_pixel * ratio_coo_pixel

    # Compute the mandelbrot set
    M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
    plot_image(M, N_coo, x1_coo, y1_coo)

    myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
    myobj.set_data(M)
    ax.add_patch(Rectangle((1 - .1, 1 - .1), 0.2, 0.2,
                           alpha=1, facecolor="none", fill=None, ))
    ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, L=%d" %
                 (N_coo, x1_coo, y1_coo, cmaps[i_cmap], L))
    plt.draw()


def zoom_on_point(event):
    global N_pixel, N_coo, x1_coo, y1_coo, myobj, L, M, i_cmap, power
    if event.button == 3 and event.inaxes:
        # Zoom on clicked point; new N_coo=10% of old N_coo
        x1_pixel, y1_pixel = event.xdata, event.ydata
        x1_coo = x1_coo+N_coo*(x1_pixel-N_pixel/2.)/N_pixel
        y1_coo = y1_coo+N_coo*(y1_pixel-N_pixel/2.)/N_pixel
        N_coo = N_coo*.1

        M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
        plot_image(M, N_coo, x1_coo, y1_coo)

        myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, L=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], L))
        plt.draw()
    if not event.inaxes and event.x < .3*N_pixel:
        # Click on left N_coo of image to reset to full fractal
        power = 2
        N_coo = 3.0
        x1_coo = -.5
        y1_coo = 0.
        i_cmap = 49

        M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
        plot_image(M, N_coo, x1_coo, y1_coo)

        myobj = plt.imshow(M, cmap=cmaps[i_cmap], origin="lower")
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, L=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], L))
        plt.draw()
    # Left click on right N_coo of image to set a random colormap
    if event.button == 1 and not event.inaxes and event.x > .7*N_pixel:
        i_cmap_current = i_cmap
        i_cmap = np.random.randint(len(cmaps))
        if i_cmap == i_cmap_current:
            i_cmap -= 1
            if i_cmap < 0:
                i_cmap = len(cmaps)-1
        myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, L=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], L))
        plt.draw()
    # Right click on right N_coo to set colormap="flag"
    if event.button == 3 and not event.inaxes and event.x > .7*N_pixel:
        i_cmap = 49
        myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, L=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], L))
        plt.draw()


zoom_on_point.RS = RectangleSelector(
    ax, zoom_on_square, drawtype="box", useblit=True, button=[1, 3],
    minspanx=5, minspany=5, spancoords="pixels")

def plot_image(M, N_coo, x1_coo, y1_coo):
    d_image = cuda.to_device(M)
    mandel_kernel[griddim, blockdim](x1_coo, y1_coo, N_coo, d_image)
    d_image.to_host()


fig.canvas.mpl_connect("button_press_event", zoom_on_point)
fig.canvas.mpl_connect("key_press_event", key_selector)

M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
plot_image(M, N_coo, x1_coo, y1_coo)

ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, L=%d" %
             (N_coo, x1_coo, y1_coo, cmaps[i_cmap], L))
plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
plt.show()
