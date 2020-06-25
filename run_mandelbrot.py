import argparse

import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from numba import cuda
from pylab import cm as cm

BLOCKDIM = (32, 8)
GRIDDIM = (32, 16)


@numba.jit
def mandel(x, y, iters):
    c = complex(x, y)
    z = 0.0j
    for i in range(iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return iters


@numba.jit
def create_fractal(x_0, y_0, n_coo, d_image, n_pixel, iters):

    pixel_size_x = n_coo / n_pixel
    pixel_size_y = n_coo / n_pixel

    for x in range(n_pixel):
        real = x_0 + x * pixel_size_x
        for y in range(n_pixel):
            imag = y_0 + y * pixel_size_y
            color = mandel(real, imag, iters)
            d_image[y, x] = color


@cuda.jit
def mandel_kernel(x_0, y_0, n_coo, d_image, n_pixel, iters):

    pixel_size_x = n_coo / n_pixel
    pixel_size_y = n_coo / n_pixel

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, n_pixel, gridX):
        real = x_0 + x * pixel_size_x
        for y in range(startY, n_pixel, gridY):
            imag = y_0 + y * pixel_size_y
            d_image[y, x] = mandel_gpu(real, imag, iters)


def zoom_on_square(eclick, erelease):
    global n_pixel, n_coo, x1_coo, y1_coo, myobj, image, power

    x1_pixel, y1_pixel = min(eclick.xdata, erelease.xdata), min(
        eclick.ydata, erelease.ydata)
    x2_pixel, y2_pixel = max(eclick.xdata, erelease.xdata), max(
        eclick.ydata, erelease.ydata)

    # Get correct ratio zoomed image
    H = (y2_pixel - y1_pixel)
    W = (x2_pixel - x1_pixel)
    ratio_h_w = H / W
    if ratio_h_w > 1:
        difference = (H - W) / 2
        y1_pixel = y1_pixel + difference
        y2_pixel = y2_pixel - difference
    elif ratio_h_w <= 1:
        difference = (W - H) / 2
        x1_pixel = x1_pixel + difference
        x2_pixel = x2_pixel - difference

    assert round(y2_pixel - y1_pixel, 2) == round(x2_pixel - x1_pixel, 2)

    zoom_N_pixel = y2_pixel - y1_pixel
    # Convert from pixel to coordinates in the plane
    ratio_coo_pixel = n_coo / n_pixel
    x1_coo = x1_coo + x1_pixel * ratio_coo_pixel
    y1_coo = y1_coo + y1_pixel * ratio_coo_pixel
    n_coo = zoom_N_pixel * ratio_coo_pixel

    # Compute the mandelbrot set
    image = np.zeros((n_pixel, n_pixel), dtype=np.uint8)
    plot_image(no_cuda, image, n_coo, x1_coo, y1_coo, n_pixel, iters)

    myobj = plt.imshow(image, origin="lower", cmap=cmaps[i_cmap])
    myobj.set_data(image)
    ax.add_patch(Rectangle((1 - .1, 1 - .1), 0.2, 0.2,
                           alpha=1, facecolor="none", fill=None, ))
    ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                 (n_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
    plt.draw()


def zoom_on_point(event):
    global n_pixel, n_coo, x1_coo, y1_coo, myobj, iters, image, i_cmap, power

    # Zoom on clicked point; new n_coo=10% of old n_coo
    if event.button == 3 and event.inaxes:
        x1_pixel, y1_pixel = event.xdata, event.ydata
        x1_coo = x1_coo+n_coo*(x1_pixel-n_pixel/2.)/n_pixel
        y1_coo = y1_coo+n_coo*(y1_pixel-n_pixel/2.)/n_pixel
        n_coo = n_coo*.1

        image = np.zeros((n_pixel, n_pixel), dtype=np.uint8)
        plot_image(no_cuda, image, n_coo, x1_coo, y1_coo, n_pixel, iters)

        myobj = plt.imshow(image, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (n_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()

    # Click on left n_coo of image to reset to full fractal
    if not event.inaxes and event.x < .3*n_pixel:
        power = 2
        n_coo = 3.0
        x1_coo = -.5
        y1_coo = 0.
        i_cmap = 49

        image = np.zeros((n_pixel, n_pixel), dtype=np.uint8)
        plot_image(no_cuda, image, n_coo, x1_coo, y1_coo, n_pixel, iters)

        myobj = plt.imshow(image, cmap=cmaps[i_cmap], origin="lower")
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (n_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()

    # Left click on right n_coo of image to set a random colormap
    if event.button == 1 and not event.inaxes and event.x > .7*n_pixel:
        i_cmap_current = i_cmap
        i_cmap = np.random.randint(len(cmaps))
        if i_cmap == i_cmap_current:
            i_cmap -= 1
            if i_cmap < 0:
                i_cmap = len(cmaps)-1
        myobj = plt.imshow(image, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (n_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()
    # Right click on right n_coo to set colormap="flag"
    if event.button == 3 and not event.inaxes and event.x > .7*n_pixel:
        i_cmap = 49
        myobj = plt.imshow(image, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (n_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()


def plot_image(no_cuda, image, n_coo, x1_coo, y1_coo, n_pixel, iters):
    if no_cuda:
        create_fractal(x1_coo, y1_coo, n_coo, image, n_pixel, iters)
    else:
        d_image = cuda.to_device(image)
        mandel_kernel[GRIDDIM, BLOCKDIM](x1_coo, y1_coo, n_coo,
                                         d_image, n_pixel, iters)
        d_image.to_host()


def main():
    global cmaps, ax, image, mandel_gpu

    if not no_cuda:
        mandel_gpu = cuda.jit(device=True)(mandel)

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Interactive Mandelbrot Set Accelerated using Numba")
    ax = fig.add_subplot(111)
    cmaps = [m for m in cm.datad if not m.endswith("_r")]

    zoom_on_point.RS = RectangleSelector(
        ax, zoom_on_square, drawtype="box", useblit=True, button=[1, 3],
        minspanx=5, minspany=5, spancoords="pixels")

    fig.canvas.mpl_connect("button_press_event", zoom_on_point)

    image = np.zeros((n_pixel, n_pixel), dtype=np.uint8)
    plot_image(no_cuda, image, n_coo, x1_coo, y1_coo, n_pixel, iters)

    ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                (n_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
    plt.imshow(image, origin="lower", cmap=cmaps[i_cmap])
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing Pipeline")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--n_pixel", type=int, default=1024)
    parser.add_argument("--x1_coo", type=float, default=-1.5)
    parser.add_argument("--y1_coo", type=float, default=-1.5)
    parser.add_argument("--n_coo", type=float, default=3.)
    parser.add_argument("--i_cmap", type=int, default=49)
    parser.add_argument("--power", type=int, default=2)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    global no_cuda, n_coo, n_pixel, x1_coo, y1_coo, iters, i_cmap, power

    args = parse_args()
    no_cuda=args.no_cuda
    n_coo = args.n_coo
    n_pixel = args.n_pixel
    x1_coo = args.x1_coo
    y1_coo = args.y1_coo
    iters = args.iters
    i_cmap = args.i_cmap
    power = args.power

    main()
