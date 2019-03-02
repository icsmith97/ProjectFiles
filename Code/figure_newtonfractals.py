from PIL import Image
import numpy as np
from multiprocessing import Pool
from itertools import product
import time


def image_gen(p):
        f = lambda x: x**8 + 15*x**4 - 16
        f_prime = lambda x: 8*x**7 + 60*x**3
        roots = []

        colors = [(255, 97, 136), (169, 220, 118), (255, 216, 102), (120, 220, 232), (252, 152, 102), (171, 157, 242),
        (246, 246, 246), (128, 128, 128)]  # select colours: Monokai Theme

        tolerance = 1  # convergence criteria
        max_its = 25  # number of iterations

        re = p[0][0]
        im = p[1][0]
        z = re+1j*im
        for i in range(max_its):
            try:
                z -= (f(z))/(f_prime(z))
            except ZeroDivisionError:
                # possibly divide by 0
                continue
            if abs(f(z)) < tolerance:
                break

        color_depth = (max_its - i) * 1 / max_its

        # find to which solution this guess converged to
        err = [abs(z-root) for root in roots]
        distances = zip(err, range(len(colors)))

        # select the color associated with the solution
        color = [int(i*color_depth) for i in colors[min(distances)[1]]]

        if p[0][1] % 500 == 0 and p[1][1] == p[0][1]:
            print("Getting there...")
        return p[0][1], p[1][1], color


if __name__=='__main__':

    real_domain = [-3, 3]
    imag_domain = [ 2, -2]

    resolution = [300, 200]           # image size

    # Program Begins

    img = Image.new("RGB", (resolution[0], resolution[1]), (0, 0, 0))

    reals = list(np.linspace(real_domain[0], real_domain[1], resolution[0]))
    imags = list(np.linspace(imag_domain[1], imag_domain[0], resolution[1]))

    re_pos = range(resolution[0])
    im_pos = range(resolution[1])

    rep = zip(reals, re_pos)
    imp = zip(imags, im_pos)

    print("Making list")
    points = list(product(rep, imp))

    print("Running Newton's Method")

    t0 = time.time()
    p = Pool()
    ims = p.map(image_gen, points)
    t1 = time.time()

    print("Parallel: " + str(t1-t0))

    for i in ims:
        img.putpixel((i[0], i[1]), tuple(i[2]))

    img.save('fractal_z4s_%03dx%03dcheck7.png' %
                 (resolution[0], resolution[1]),
                 dpi=(150, 150))
