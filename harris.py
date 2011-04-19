# -*- encoding: utf8 -*-
import math
import numpy
from scipy import signal
from PIL import Image, ImageDraw, ImageOps, ImageTk
from collections import defaultdict


def dist_l2(p1, p2):
    return sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]) ** (1/2.)

def gaussian_mask(width, height, sigma):
    u = (width - 1) / 2
    v = (height - 1) / 2

    w = []
    for i in xrange(-u, u + 1):
        w.append([math.exp(-(i**2 + j**2)/(2*(sigma**2)))
            for j in xrange(-v, v + 1)])
    return w

def flood_fill(pix, max_x, max_y, color=128):
    to_do = [(0, 0)]
    while to_do:
        x, y = to_do.pop()
        pix[x, y] = color
        if x > 0:
            if not pix[x-1, y]: to_do.append((x-1, y))
        if x < max_x - 1:
            if not pix[x+1, y]: to_do.append((x+1, y))
        if y < max_y - 1:
            if not pix[x, y+1]: to_do.append((x, y+1))
        if y > 0:
            if not pix[x, y-1]: to_do.append((x, y-1))


def harris(img, threshold, sigma=0.5, wwidth=3, wheight=3, invert=False):
    grayscale = False
    if len(img.getbands()) == 1:
        if img.getbands()[0] == 'L':
            grayscale = True
        img = img.convert('RGB')
    if invert:
        img = ImageOps.invert(img)

    new_img = img.copy()
    img_pix = new_img.load()

    # R matrix with corner points as 8-way local maximum and
    # non-maximum suppression.
    raw_harris_img = Image.new('RGB', img.size)
    # Discard corner points above a threshold
    thr1_img = Image.new('RGB', img.size)
    # Discard edge points above an upper threshold
    uthr_img = Image.new('RGB', img.size)
    # Hysteresis
    hyst_img = Image.new('RGB', img.size)
    # Fill gaps in both x and y
    gapsxy_img = None
    # Caption without median filter
    almost_there = None

    edge_img = Image.new('L', img.size)
    edge_pix = edge_img.load()
    corner_img = Image.new('L', img.size)
    corner_pix = corner_img.load()
    caption_reg_img = Image.new('L', img.size)
    caption_pix = caption_reg_img.load()

    w = gaussian_mask(wwidth, wheight, sigma)

    #band = list(img.split()) + [ImageOps.grayscale(img)]
    #band = [ImageOps.grayscale(img)]
    band = list(img.split())
    if grayscale:
        band = [band[0]]
    for img_gray in band:
        do_harris(new_img, img_gray, raw_harris_img, thr1_img, uthr_img,
                hyst_img, edge_pix, edge_img, corner_pix, threshold, w)

    max_x, max_y = new_img.size

    # Try to fill gaps both in horizontal and vertical
    new_white = []
    for x in xrange(max_x):
        for y in xrange(max_y):
            if edge_pix[x, y]:
                continue
            try:
                if edge_pix[x-1, y] or edge_pix[x+1, y]:
                    new_white.append((x, y))
                elif edge_pix[x, y-1] or edge_pix[x, y+1]:
                    new_white.append((x, y))
            except IndexError:
                pass
    for x, y in new_white:
        edge_pix[x, y] = 255
    gaps_img = edge_img.copy()

    flood_fill(edge_pix, max_x, max_y, color=128)

    for x in xrange(max_x):
        for y in xrange(max_y):
            if edge_pix[x, y] or caption_pix[x, y]:
                continue
            caption_pix[x, y] = 255
            #x_orig, y_orig = x, y
            #while y - 1 > 0 and edge_pix[x, y] in (255, 0):
            while y - 1 > 0 and not edge_pix[x, y]:
                y -= 1
                caption_pix[x, y] = 255
            #x, y = x_orig, y_orig
            #while y + 1 < max_y and edge_pix[x, y] in (255, 0):
            while y + 1 < max_y and not edge_pix[x, y]:
                y += 1
                caption_pix[x, y] = 255
            #x, y = x_orig, y_orig
            #while x + 1 < max_x and edge_pix[x, y] in (255, 0):
            while x + 1 < max_x and not edge_pix[x, y]:
                x += 1
                caption_pix[x, y] = 255
            #x, y = x_orig, y_orig
            #while x - 1 > 0 and edge_pix[x, y] in (255, 0):
            while x - 1 > 0 and not edge_pix[x, y]:
                x -= 1
                caption_pix[x, y] = 255

    almost_there = caption_reg_img.copy()

    # Median filter
    data = list(caption_reg_img.getdata())
    masksize = 5
    maskelems = masksize ** 2
    start = (masksize - 1) / 2

    for x in xrange(start, max_x - start):
        for y in xrange(start, max_y - start):
            val = [caption_pix[x+i, y+j]
                for i in xrange(-start, start + 1)
                for j in xrange(-start, start + 1)]
            val.sort()
            data[y * max_x + x] = val[maskelems / 2]
    caption_reg_img.putdata(data)


    return (w, new_img, raw_harris_img, thr1_img, uthr_img, hyst_img,
            gaps_img, almost_there, edge_img, caption_reg_img)
            #gaps_img, almost_there, edge_img, corner_img, caption_reg_img)


def do_harris(new_img, img_gray, raw_harris_img, thr1_img, uthr_img, hyst_img,
        edge_pix, edge_img, corner_pix, threshold, w):

    img_orig = new_img.copy()
    img_pix = new_img.load()
    img2d = numpy.asarray(img_gray)

    X = signal.convolve2d(img2d, [[-1, 0, 1]], mode='same')
    Y = signal.convolve2d(img2d, [[-1], [0], [1]], mode='same')

    A = signal.convolve2d(X ** 2, w, mode='same')
    B = signal.convolve2d(Y ** 2, w, mode='same')
    C = signal.convolve2d(X * Y, w, mode='same')

    Tr = A + B
    Det = A * B - (C ** 2)
    k = 0.05
    R = Det - k * (Tr ** 2)

    print '-+' * 30

    max_x, max_y = new_img.size

    # Corner points
    corner = []
    max_interest = numpy.max(R)
    discard = (max_interest / 100.) * threshold

    # Hysteresis on edges
    edge = set()
    maybe_edge = defaultdict(int)
    max_edge = abs(numpy.min(R))
    #min_edge = abs(numpy.max(R[R<0]))
    #avg_edge = abs(numpy.mean(R[R<0]))
    #stddev_edge = numpy.std(R[R<0])
    #median_edge = abs(numpy.median(R[R<0]))
    #upper_threshold = max_edge - (max_edge / 100. * 98)
    upper_threshold = max_edge / 100. * 2
    #lower_threshold = max_edge - (max_edge / 100. * 99.99)
    lower_threshold = max_edge / 100. * 0.01
    print "Hysteresis thresholds: %.2f %.2f" % (upper_threshold,
            lower_threshold)

    xstart = (len(w[0]) + 1) / 2
    ystart = (len(w) + 1) / 2

    raw_pix = raw_harris_img.load()
    raw_draw = ImageDraw.Draw(raw_harris_img)
    thr1_pix = thr1_img.load()
    thr1_draw = ImageDraw.Draw(thr1_img)
    uthr_pix = uthr_img.load()
    uthr_draw = ImageDraw.Draw(uthr_img)
    hyst_pix = hyst_img.load()
    hyst_draw = ImageDraw.Draw(hyst_img)
    edge_color = (0, 128, 255)
    corner_color = (255, 255, 255)

    for i in xrange(xstart, max_x - xstart):
        for j in xrange(ystart, max_y - ystart):
            # Corner pixel:
            # Check if R[j][i] is an 8-way local maximum
            if R[j, i] > 0:
                if R[j, i] == numpy.max(R[j-1:j+2, i-1:i+2]):
                    raw_draw.rectangle((i-1, j-1, i+1, j+1), fill=corner_color)
                    #if R[j, i] - discard > 0:
                    if R[j, i] > discard:
                        corner.append([i, j])

            # Edge pixel
            elif R[j, i] < 0:
                test_edge = None
                # XXX Descobri que isso é chamado de "Non-Maximum Suppression"
                if A[j, i] > B[j, i]:
                    if R[j, i] < min([R[j, i - 1], R[j, i + 1]]):
                        test_edge = (i, j, abs(R[j, i]))
                        raw_pix[i, j] = edge_color
                        thr1_pix[i, j] = edge_color
                else:
                    if R[j, i] < min([R[j - 1, i], R[j + 1, i]]):
                        test_edge = (i, j, abs(R[j, i]))
                        raw_pix[i, j] = edge_color
                        thr1_pix[i, j] = edge_color

                if test_edge is not None:
                    #img_pix[test_edge[0], test_edge[1]] = (0, 0, 255)
                    #edge_pix[test_edge[0], test_edge[1]] = 255
                    if test_edge[2] > upper_threshold:
                        #edge.append([test_edge[0], test_edge[1]])
                        edge.add((test_edge[0], test_edge[1]))
                        uthr_pix[test_edge[0], test_edge[1]] = edge_color
                        maybe_edge[(test_edge[0], test_edge[1])] = 1
                    elif test_edge[2] > lower_threshold:
                        maybe_edge[(test_edge[0], test_edge[1])] = 1

    # Mark and draw corner points.
    draw = ImageDraw.Draw(new_img)
    for (i, j) in corner:
        corner_pix[i, j] = 255
        draw.rectangle((i-1, j-1, i+1, j+1), fill='red')
        thr1_draw.rectangle((i-1, j-1, i+1, j+1), fill=corner_color)
        hyst_draw.rectangle((i-1, j-1, i+1, j+1), fill=corner_color)
        uthr_draw.rectangle((i-1, j-1, i+1, j+1), fill=corner_color)

    do_hysteresis(img_pix, edge_pix, edge, maybe_edge,lower_threshold,hyst_pix)

    draw = ImageDraw.Draw(edge_img)
    for x, y in corner:
        draw.rectangle((x-1, y-1, x+1, y+1), fill='white')

    #return R


# XXX Testing hysteresis
def do_hysteresis(img_pix, edge_pix, edge, maybe_edge,lower_threshold,hyst_pix):
    nhood = 3
    dirs = [(x, y) for x in xrange(-nhood, nhood + 1)
       for y in xrange(-nhood, nhood + 1) if x != 0 and y != 0]

    while edge:
        i, j = edge.pop()
        img_pix[i, j] = (0, 255, 255)
        edge_pix[i, j] = 255
        hyst_pix[i, j] = (0, 128, 255)
        for x, y in dirs:
            try:
                if (maybe_edge[i+x, j+y] and not edge_pix[i+x, j+y]):
                    edge.add((i+x, j+y))
            except IndexError:
                pass
