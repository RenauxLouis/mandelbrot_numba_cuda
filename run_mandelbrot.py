# Interactive Mandelbrot Set Accelerated using Numba
# Classical Iteration Method
# Luis Villasenor
# lvillasen@gmail.com
# 2/8/2016
# Licence: GPLv3

# Usage

# Use the left buttom to draw a square to zoom into

# Point and click with the right buttom to magnify by a factor of 10

# Click with the left button on the rigth side of the
# image to randomly change the colormap

# Click with right button on the right side of the image to set the default colormap

# Click on the left side of the image to restart with the full Mandelbrot set

# Press the up/down arrow to increase/decrease the maximum number of iterations

# Press the right/left arrow to increase/decrease the number of pixels

# Type a number from 1-9 to set power index of the iteration formula

# Type 'f' to toggle full-screen mode

# Type 's' to save the image

import numpy as np
from pylab import cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import numba
from numba import cuda


global N_pixel, x1_coo, y1_coo, N_coo, L, M, power
L=800
N_pixel=1024
ratio_image = 1.
x1_coo=-1.5
y1_coo=-1.5
N_coo=3.0
i_cmap=49
power=2
fig = plt.figure(figsize=(12,12))
fig.suptitle('Interactive Mandelbrot Set Accelerated using Numba')
ax = fig.add_subplot(111)
cmaps=[m for m in cm.datad if not m.endswith("_r")]

blockdim = (32, 8)
griddim = (32,16)


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
# def mandel_kernel(min_x, max_x, min_y, max_y, image):
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
    'eclick and erelease are the press and release events'
    global N_pixel, N_coo, x1_coo, y1_coo, myobj, M, power

    x1_pixel, y1_pixel = min(eclick.xdata,erelease.xdata), min(eclick.ydata,erelease.ydata)
    x2_pixel, y2_pixel = max(eclick.xdata,erelease.xdata), max(eclick.ydata,erelease.ydata)

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
    M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
    plot_image(M)
    myobj = plt.imshow(M,origin='lower',cmap=cmaps[i_cmap])
    myobj.set_data(M)
    ax.add_patch(Rectangle((1 - .1, 1 - .1), 0.2, 0.2,alpha=1, facecolor='none',fill=None, ))
    ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
    plt.draw()


def key_selector(event):
    global N_pixel,N_coo,x1_coo,y1_coo,myobj,M,power,L,i_cmap
    #print(' Key pressed.')
    if event.key == u'up':  # Increase max number of iterations
        L=int(L*1.2);

        M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
        # create_fractal(x1_coo, y1_coo, N_coo, M)
        plot_image(M)
        # M = np.zeros((N, N), dtype = np.uint8)
        # d_image = cuda.to_device(M)
        # mandel_kernel[griddim, blockdim](x1_coo,y1_coo,x1_coo+N_coo,y1_coo+L, d_image)
        # d_image.to_host()

        myobj = plt.imshow(M,cmap=cmaps[i_cmap],origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if event.key == u'down':  # Decrease max number of iterations
        L=int(L/1.2);


        # M = np.zeros((1024, 1536), dtype = np.uint8)
        # d_image = cuda.to_device(M)
        # mandel_kernel[griddim, blockdim](x1_coo,y1_coo,x1_coo+N_coo,y1_coo+L, d_image)
        # d_image.to_host()
        M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
        # create_fractal(x1_coo, y1_coo, N_coo, M)
        plot_image(M)

        myobj = plt.imshow(M,cmap=cmaps[i_cmap],origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if event.key == u'right':  # Increase  number of pixels
        N_pixel=int(N_pixel*1.2);


        # M = np.zeros((N, N), dtype = np.uint8)
        # d_image = cuda.to_device(M)
        # mandel_kernel[griddim, blockdim](x1_coo,y1_coo,x1_coo+N_coo,y1_coo+L, d_image)
        # d_image.to_host()

        M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
        # create_fractal(x1_coo, y1_coo, N_coo, M)
        plot_image(M)

        myobj = plt.imshow(M,cmap=cmaps[i_cmap],origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if event.key == u'left':  # Decrease  number of pixels
        N_pixel=int(N_pixel/1.2);


        # M = np.zeros((N, N), dtype = np.uint8)
        # d_image = cuda.to_device(M)
        # mandel_kernel[griddim, blockdim](x1_coo,y1_coo,x1_coo+N_coo,y1_coo+L, d_image)
        # d_image.to_host()

        M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
        # create_fractal(x1_coo, y1_coo, N_coo, M)
        plot_image(M)

        myobj = plt.imshow(M,cmap=cmaps[i_cmap],origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if event.key in ['1','2','3','4','5','6','7','8','9'] :  # Decrease  number of pixels
        power=int(event.key)
        if power <10 and power >0 :
            print("Power index set to %d" % power)
            i_cmap=49
            N_coo=3.0; x1_coo=-.5;y1_coo=0.;L=200;

            M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
            # create_fractal(x1_coo, y1_coo, N_coo, M)
            plot_image(M)

            # M = np.zeros((N, N), dtype = np.uint8)
            # d_image = cuda.to_device(M)
            # mandel_kernel[griddim, blockdim](x1_coo,y1_coo,x1_coo+N_coo,y1_coo+L, d_image)
            # d_image.to_host()

            myobj = plt.imshow(M,cmap=cmaps[i_cmap],origin='lower')
            ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
            plt.draw()



key_selector.RS = RectangleSelector(ax, zoom_on_square,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels')

def zoom_on_point(event):
    global N_pixel,N_coo,x1_coo,y1_coo,myobj,L,M,i_cmap,power
    #print(" Button pressed: %d" % (event.button))
    #print(' event.x= %f, event.y= %f '%(event.x,event.y))
    if event.button==3 and event.inaxes: # Zoom on clicked point; new N_coo=10% of old N_coo
        x1_pixel, y1_pixel = event.xdata, event.ydata
        x1_coo=x1_coo+N_coo*(x1_pixel-N_pixel/2.)/N
        y1_coo=y1_coo+N_coo*(y1_pixel-N_pixel/2.)/N
        N_coo=N_coo*.1

        M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
        plot_image(M)

        myobj = plt.imshow(M,origin='lower',cmap=cmaps[i_cmap])
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if not event.inaxes and event.x<.3*N_pixel : # Click on left N_coo of image to reset to full fractal
        power=2; N_coo=3.0; x1_coo=-.5;y1_coo=0.;i_cmap=49

        M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
        plot_image()

        myobj = plt.imshow(M,cmap=cmaps[i_cmap],origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if event.button==1 and not event.inaxes and event.x>.7*N_pixel : # Left click on right N_coo of image to set a random colormap
        i_cmap_current=i_cmap
        i_cmap=np.random.randint(len(cmaps))
        if i_cmap==i_cmap_current:
            i_cmap-=1
            if i_cmap< 0 : i_cmap=len(cmaps)-1
        myobj = plt.imshow(M,origin='lower',cmap=cmaps[i_cmap])
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()
    if event.button==3 and not event.inaxes and event.x>.7*N_pixel : # Right click on right N_coo to set mapolormap='flag'
        i_cmap=49
        myobj = plt.imshow(M,origin='lower',cmap=cmaps[i_cmap])
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
        plt.draw()

def plot_image(M):
    global N_coo,x1_coo,y1_coo

    d_image = cuda.to_device(M)
    mandel_kernel[griddim, blockdim](x1_coo, y1_coo, N_coo, d_image)
    d_image.to_host()

fig.canvas.mpl_connect('button_press_event', zoom_on_point)
fig.canvas.mpl_connect('key_press_event', key_selector)

M = np.zeros((N_pixel, N_pixel), dtype = np.uint8)
plot_image(M)

ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, L=%d'%(N_coo,x1_coo,y1_coo,cmaps[i_cmap],L))
plt.imshow(M,origin='lower',cmap=cmaps[i_cmap])
plt.show()

