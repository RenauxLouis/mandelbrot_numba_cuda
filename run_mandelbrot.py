import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from numba import cuda
from pylab import cm as cm
import argparse

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


mandel_gpu = cuda.jit(device=True)(mandel)


@cuda.jit
def mandel_kernel(x_0, y_0, N_coo, d_image, N_pixel, iters):
    pixel_size_x = N_coo / N_pixel
    pixel_size_y = N_coo / N_pixel

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, N_pixel, gridX):
        real = x_0 + x * pixel_size_x
        for y in range(startY, N_pixel, gridY):
            imag = y_0 + y * pixel_size_y
            d_image[y, x] = mandel_gpu(real, imag, iters)


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
    ratio_coo_pixel = N_coo / N_pixel
    x1_coo = x1_coo + x1_pixel * ratio_coo_pixel
    y1_coo = y1_coo + y1_pixel * ratio_coo_pixel
    N_coo = zoom_N_pixel * ratio_coo_pixel

    # Compute the mandelbrot set
    M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
    plot_image(M, N_coo, x1_coo, y1_coo, N_pixel, iters)

    myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
    myobj.set_data(M)
    ax.add_patch(Rectangle((1 - .1, 1 - .1), 0.2, 0.2,
                           alpha=1, facecolor="none", fill=None, ))
    ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                 (N_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
    plt.draw()


def zoom_on_point(event):
    global N_pixel, N_coo, x1_coo, y1_coo, myobj, iters, M, i_cmap, power

    # Zoom on clicked point; new N_coo=10% of old N_coo
    if event.button == 3 and event.inaxes:
        x1_pixel, y1_pixel = event.xdata, event.ydata
        x1_coo = x1_coo+N_coo*(x1_pixel-N_pixel/2.)/N_pixel
        y1_coo = y1_coo+N_coo*(y1_pixel-N_pixel/2.)/N_pixel
        N_coo = N_coo*.1

        M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
        plot_image(M, N_coo, x1_coo, y1_coo, N_pixel, iters)

        myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()

    # Click on left N_coo of image to reset to full fractal
    if not event.inaxes and event.x < .3*N_pixel:
        power = 2
        N_coo = 3.0
        x1_coo = -.5
        y1_coo = 0.
        i_cmap = 49

        M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
        plot_image(M, N_coo, x1_coo, y1_coo, N_pixel, iters)

        myobj = plt.imshow(M, cmap=cmaps[i_cmap], origin="lower")
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
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
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()
    # Right click on right N_coo to set colormap="flag"
    if event.button == 3 and not event.inaxes and event.x > .7*N_pixel:
        i_cmap = 49
        myobj = plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
        ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                     (N_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
        plt.draw()


def plot_image(M, N_coo, x1_coo, y1_coo, N_pixel, iters):
    d_image = cuda.to_device(M)
    mandel_kernel[GRIDDIM, BLOCKDIM](x1_coo, y1_coo, N_coo, d_image, N_pixel, iters)
    d_image.to_host()


def main(iters, N_pixel, x1_coo, y1_coo, N_coo, i_cmap, power):
    global cmaps, ax

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Interactive Mandelbrot Set Accelerated using Numba")
    ax = fig.add_subplot(111)
    cmaps = [m for m in cm.datad if not m.endswith("_r")]

    zoom_on_point.RS = RectangleSelector(
        ax, zoom_on_square, drawtype="box", useblit=True, button=[1, 3],
        minspanx=5, minspany=5, spancoords="pixels")

    fig.canvas.mpl_connect("button_press_event", zoom_on_point)

    M = np.zeros((N_pixel, N_pixel), dtype=np.uint8)
    plot_image(M, N_coo, x1_coo, y1_coo, N_pixel, iters)

    ax.set_title("Side=%.2e, x=%.2e, y=%.2e, %s, iters=%d" %
                (N_coo, x1_coo, y1_coo, cmaps[i_cmap], iters))
    plt.imshow(M, origin="lower", cmap=cmaps[i_cmap])
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing Pipeline")
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--N_pixel", type=int, default=1024)
    parser.add_argument("--x1_coo", type=float, default=-1.5)
    parser.add_argument("--y1_coo", type=float, default=-1.5)
    parser.add_argument("--N_coo", type=float, default=3.)
    parser.add_argument("--i_cmap", type=int, default=49)
    parser.add_argument("--power", type=int, default=2)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    global N_coo, N_pixel, x1_coo, y1_coo, iters

    args = parse_args()
    N_coo = args.N_coo
    N_pixel = args.N_pixel
    x1_coo = args.x1_coo
    y1_coo = args.y1_coo
    iters = args.iters
    i_cmap = args.i_cmap
    power = args.power

    main(iters=iters, N_pixel=N_pixel,
         x1_coo=x1_coo, y1_coo=y1_coo, N_coo=N_coo,
         i_cmap=i_cmap, power=power)
