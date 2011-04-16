# -*- encoding: utf8 -*-
import os
import sys
import math
import time
import numpy
import Tkinter
import tkFileDialog
from scipy import signal
from PIL import Image, ImageDraw, ImageOps, ImageTk
from colorsys import rgb_to_hsv
from collections import defaultdict

sys.setrecursionlimit(10000)

#import matplotlib
#matplotlib.use('TkAgg') # XXX Esse backend deixa mais lento o plot,
#                        # mas se não usar não da pra mudar de imagem :/
#
#from matplotlib import pyplot
#from matplotlib.colors import BoundaryNorm

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

def harris(img, threshold, sigma=0.5, wwidth=3, wheight=3, invert=False):
    if len(img.getbands()) == 1:
        img = img.convert('RGB')
    if invert:
        img = ImageOps.invert(img)

    new_img = img.copy()
    img_pix = new_img.load()

    edge_img = Image.new('L', img.size)
    edge_pix = edge_img.load()
    corner_img = Image.new('L', img.size)
    corner_pix = corner_img.load()
    #rect_img = Image.new('L', img.size)
    caption_reg_img = Image.new('L', img.size)
    caption_pix = caption_reg_img.load()

    w = gaussian_mask(wwidth, wheight, sigma)

    #band = list(img.split()) + [ImageOps.grayscale(img)]
    #band = [ImageOps.grayscale(img)]
    band = list(img.split())
    for img_gray in band:
        do_harris(new_img, img_gray, edge_pix, edge_img, corner_pix, #rect_img,
                threshold, w)
        from scipy import ndimage
        edge_blur = edge_img.copy()
        edge_blur_pix = edge_blur.load()
        result = ndimage.gaussian_filter(numpy.asarray(edge_blur), 1)
        for x in xrange(result.shape[1]):
            for y in xrange(result.shape[0]):
                edge_blur_pix[x, y] = result[y, x]

    max_x, max_y = new_img.size

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

    flood_fill(edge_pix, max_x, max_y)

    for x in xrange(max_x):
        for y in xrange(max_y):
            if edge_pix[x, y] or caption_pix[x, y]:
                continue
            caption_pix[x, y] = 255
            while y - 1 > 0 and not not edge_pix[x, y]:# in (255, 0):
                y -= 1
                caption_pix[x, y] = 255
            while y + 1 < max_y and not edge_pix[x, y]:# in (255, 0):
                y += 1
                caption_pix[x, y] = 255
            while x + 1 < max_x and not edge_pix[x, y]:# in (255, 0):
                x += 1
                caption_pix[x, y] = 255
            while x - 1 > 0 and not edge_pix[x, y]:# in (255, 0):
                x -= 1
                caption_pix[x, y] = 255

    data = list(caption_reg_img.getdata())
    masksize = 3
    maskelems = masksize ** 2
    start = (masksize - 1) / 2

    for _ in xrange(2): # XXX N aplicações do filtro da mediana
        for x in xrange(start, max_x - start):
            for y in xrange(start, max_y - start):
                val = [caption_pix[x+i, y+j]
                    for i in xrange(-start, start + 1)
                    for j in xrange(-start, start + 1)]
                val.sort()
                data[y * max_x + x] = val[maskelems / 2]
    caption_reg_img.putdata(data)


    return w, new_img, edge_img, corner_img, edge_blur, caption_reg_img


#def do_harris(new_img, img_gray, edge_pix, corner_pix, rect_img, threshold, w):
def do_harris(new_img, img_gray, edge_pix, edge_img,corner_pix, threshold, w):
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
    edge = set()#[]
    maybe_edge = defaultdict(int)
    max_edge = abs(numpy.min(R[R<0]))
    #min_edge = abs(numpy.max(R[R<0]))
    #avg_edge = abs(numpy.mean(R[R<0]))
    #stddev_edge = numpy.std(R[R<0])
    #median_edge = abs(numpy.median(R[R<0]))
    upper_threshold = max_edge - (max_edge / 100. * 99)
    lower_threshold = max_edge - (max_edge / 100. * 99.999)
    print "Hysteresis thresholds: %.2f %.2f" % (upper_threshold,
            lower_threshold)

    xstart = (len(w[0]) + 1) / 2
    ystart = (len(w) + 1) / 2

    for i in xrange(xstart, max_x - xstart):
        for j in xrange(ystart, max_y - ystart):
            # Corner pixel:
            # Check if R[j][i] is an 8-way local maximum
            if R[j, i] > 0 and R[j, i] == numpy.max(R[j-1:j+2, i-1:i+2]):
                if R[j, i] - discard > 0:
                    corner.append([i, j])

            # Edge pixel
            elif R[j, i] < 0:
                test_edge = None
                # XXX Descobri que isso é chamado de "Non-Maximum Suppression"
                if A[j, i] > B[j, i]:
                    if R[j, i] < min([R[j, i - 1], R[j, i + 1]]):
                        test_edge = (i, j, abs(R[j, i]))
                else:
                    if R[j][i] < min([R[j - 1, i], R[j + 1, i]]):
                        test_edge = (i, j, abs(R[j, i]))

                if test_edge is not None:
                    #img_pix[test_edge[0], test_edge[1]] = (0, 0, 255)
                    #edge_pix[test_edge[0], test_edge[1]] = 255
                    if test_edge[2] > upper_threshold:
                        #edge.append([test_edge[0], test_edge[1]])
                        edge.add((test_edge[0], test_edge[1]))
                    elif test_edge[2] > lower_threshold:
                        maybe_edge[(test_edge[0], test_edge[1])] = 1

    # Mark and draw corner points.
    draw = ImageDraw.Draw(new_img)
    for (i, j) in corner:
        corner_pix[i, j] = 255
        draw.rectangle((i-1, j-1, i+1, j+1), fill='red')

    nhood = 1
    dirs = [(x, y) for x in xrange(-nhood, nhood + 1)
       for y in xrange(-nhood, nhood + 1) if x != 0 and y != 0]
    do_hysteresis(img_pix, edge_pix, edge, maybe_edge, lower_threshold, dirs)

    draw = ImageDraw.Draw(edge_img)
    for x, y in corner:
        draw.rectangle((x-1, y-1, x+1, y+1), fill='white')

    max_x, max_y = img_orig.size
    #print dirs

    #find_rectangles(img_orig, rect_img, edge_pix, corner)

    print R
    # XXX Trying to display R as a heatmap
    #f = pyplot.figure(figsize=(4, 3), frameon=False)

    #bdata = list(numpy.linspace(R.min(), numpy.max(R[R<0]), 12)) + [0]
    #bdata += list(numpy.linspace(numpy.min(R[R>0]), R.max(), 12))

    #ax = f.add_axes([0.01, 0.01, 0.87, 0.95])
    #ax.set_axis_off()

    #im = ax.matshow(R, norm=BoundaryNorm(bdata, ncolors=256, clip=False),
    #        cmap=pyplot.cm.gnuplot2)
    #f.colorbar(im, shrink=0.89, format='%0.2g')

    #f.show()

    #from matplotlib.backends.backend_pdf import PdfPages
    #pp = PdfPages('plot1.pdf')
    #pp.savefig()
    #pp.close()


def flood_fill(edge_pix, max_x, max_y):#, x=0, y=0):
    to_do = [(0, 0)]
    while to_do:
        x, y = to_do.pop()
        edge_pix[x, y] = 128
        if x > 0:
            if not edge_pix[x-1, y]: to_do.append((x-1, y))
        if x < max_x - 1:
            if not edge_pix[x+1, y]: to_do.append((x+1, y))
        if y < max_y - 1:
            if not edge_pix[x, y+1]: to_do.append((x, y+1))
        if y > 0:
            if not edge_pix[x, y-1]: to_do.append((x, y-1))


def find_rectangles(img_orig, rect_img, edge_pix, corner):
    max_x, max_y = img_orig.size
    area = 3

    # Idéia da cabeça de ontem a noite
    print corner



def find_rectangles2(img_orig, rect_img, edge_pix, corner):
    img_pix = img_orig.load()
    _, max_y = img_orig.size

    # Transform points into cartesian coords (easier to visualize)
    interest2 = map(lambda p: [p[0], max_y - p[1]], corner)
    interest2.sort()

    test_x = defaultdict(list)
    test_y = defaultdict(list)
    max_pt_disalignment = 6
    max_cdist = 0.42 # 42 # :)

    draw = ImageDraw.Draw(rect_img)

    for (x, y) in interest2:
        for key in test_x:
            if abs(y - key) < max_pt_disalignment:
                y = key
                break
        test_x[y].append(x)

    for (x, y) in interest2:
        for key in test_y:
            if abs(x - key) < max_pt_disalignment:
                x = key
                break
        test_y[x].append(y)

    for key, value in test_y.iteritems():
        if len(value) < 2:
            continue
        print key, value
        first_color = img_pix[key, max_y - value[0]]
        value.sort()
        for v1, v2 in zip(value, value[1::]):
            for j in xrange(v1, v2):
                a = rgb_to_hsv(*img_pix[key, max_y - j])[:2]
                b = rgb_to_hsv(*first_color)[:2]
                if dist_l2(a, b) > max_cdist:
                    break
            else:
                draw.line((key, max_y - v1, key, max_y - v2), fill=255)
    print '-' * 30

    for key, value in test_x.iteritems():
        if len(value) < 2:
            continue
        print key, value
        first_color = img_pix[value[0], max_y - key]
        value.sort()
        for v1, v2 in zip(value, value[1::]):
            for i in xrange(v1, v2):
                a = rgb_to_hsv(*img_pix[i, max_y - key])[:2]
                b = rgb_to_hsv(*first_color)[:2]
                if dist_l2(a, b) > max_cdist:
                    break
            else:
                draw.line((v1, max_y - key, v2, max_y - key), fill=255)


# XXX Testing hysteresis
#def do_hysteresis(R, img_pix, edge_pix, edge, lower_threshold, dirs):
def do_hysteresis(img_pix, edge_pix, edge, maybe_edge, lower_threshold, dirs):
    while edge:
        i, j = edge.pop()
        img_pix[i, j] = (0, 255, 255)
        edge_pix[i, j] = 255
        for x, y in dirs:
            try:
                # XXX Assim não considera "Maximum Non-Suppression" feito
                #if (R[j+y, i+x] < -lower_threshold and not edge_pix[i+x, j+y]):
                #    edge.add((i+x, j+y))
                if (maybe_edge[i+x, j+y] and not edge_pix[i+x, j+y]):
                    edge.add((i+x, j+y))
            except IndexError:
                pass



def scrolledcanvas(root, master, **kwargs):
    canvas = Tkinter.Canvas(master, **kwargs)
    sx = Tkinter.Scrollbar(master, orient='horizontal')
    sy = Tkinter.Scrollbar(master, orient='vertical')
    canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
    sx['command'] = canvas.xview
    sy['command'] = canvas.yview

    sx.pack(side='bottom', fill='x')
    sy.pack(side='right', fill='y')

    root.bind('<MouseWheel>', lambda evt:
            root.nametowidget(evt.widget).yview_scroll(-evt.delta, 'units'))
    root.bind('<Shift-MouseWheel>', lambda evt:
            root.nametowidget(evt.widget).xview_scroll(-evt.delta, 'units'))

    return canvas

def canvas_addimage(canvas, img, width, height):
    canvas.create_image(0, 0, anchor='nw', image=img)
    canvas.configure(scrollregion=(0, 0, width, height),
            width=width, height=height)

class App(object):
    def __init__(self, root):
        self._build(root)

    def set_image(self, imgpath):
        src_img = Image.open(imgpath)
        img = ImageTk.PhotoImage(src_img)
        canvas_addimage(self._img_show, img, *src_img.size)
        self._img_show.s_img = img
        self._img_show.img = src_img

    def _pre_detector(self, event=None):
        try:
            threshold = float(self._threshold.get())
            sigma = float(self._sigma.get())
            wwidth = int(self._wwidth.get())
            wheight = int(self._wheight.get())
        except (TypeError, ValueError):
            raise

        if self._img_show.img is None:
            return

        invert_img = self._invert.get()

        start = time.time()
        #mask, new_img, edge_img, crn_img, rect = harris(self._img_show.img,
        mask, new_img, edge_img, crn_img, blur, result = harris(
                self._img_show.img, threshold, sigma, wwidth, wheight,
                invert=invert_img)
        #print time.time() - start
        if new_img is None:
            return
        new_img = ImageTk.PhotoImage(new_img)
        mimg = Image.new('L', (len(mask), len(mask[0])))
        mpix = mimg.load()
        for i in xrange(len(mask)):
            for j in xrange(len(mask[0])):
                mpix[i, j] = 255 * mask[i][j]
        mimg = ImageTk.PhotoImage(mimg)

        window = Tkinter.Toplevel()
        label = Tkinter.Label(window, text='%s, %s' % (threshold, sigma))
        mask_img = Tkinter.Label(window, image=mimg)
        mask_img.img = mimg
        img_frame = Tkinter.Frame(window)
        img = scrolledcanvas(window, img_frame, highlightthickness=0)
        canvas_addimage(img, new_img, new_img.width(), new_img.height())
        img.img = new_img
        label.pack()
        mask_img.pack()
        img.pack(fill='both', expand=True)
        img_frame.pack(fill='both', expand=True)

        win2 = Tkinter.Toplevel()
        eimg = ImageTk.PhotoImage(edge_img)
        eframe = Tkinter.Frame(win2)
        elbl = scrolledcanvas(win2, eframe, highlightthickness=0)
        canvas_addimage(elbl, eimg, *edge_img.size)
        elbl.img = eimg
        elbl.pack(fill='both', expand=True)
        eframe.pack(fill='both', expand=True)

        win3 = Tkinter.Toplevel()
        cimg = ImageTk.PhotoImage(crn_img)
        cframe = Tkinter.Frame(win3)
        clbl = scrolledcanvas(win3, cframe, highlightthickness=0)
        canvas_addimage(clbl, cimg, *crn_img.size)
        clbl.img = cimg
        clbl.pack(fill='both', expand=True)
        cframe.pack(fill='both', expand=True)

        #win4 = Tkinter.Toplevel()
        #rimg = ImageTk.PhotoImage(rect)
        #rframe = Tkinter.Frame(win4)
        #rlbl = scrolledcanvas(win4, rframe, highlightthickness=0)
        #canvas_addimage(rlbl, rimg, *rect.size)
        #rlbl.img = rimg
        #rlbl.pack(fill='both', expand=True)
        #rframe.pack(fill='both', expand=True)

        win3 = Tkinter.Toplevel()
        cimg = ImageTk.PhotoImage(blur)
        cframe = Tkinter.Frame(win3)
        clbl = scrolledcanvas(win3, cframe, highlightthickness=0)
        canvas_addimage(clbl, cimg, *blur.size)
        clbl.img = cimg
        clbl.pack(fill='both', expand=True)
        cframe.pack(fill='both', expand=True)

        win3 = Tkinter.Toplevel()
        cimg = ImageTk.PhotoImage(result)
        cframe = Tkinter.Frame(win3)
        clbl = scrolledcanvas(win3, cframe, highlightthickness=0)
        canvas_addimage(clbl, cimg, *result.size)
        clbl.img = cimg
        clbl.pack(fill='both', expand=True)
        cframe.pack(fill='both', expand=True)

    def _load_img(self):
        name = tkFileDialog.askopenfilename(initialdir=os.getcwd())
        if name is not None and len(name.strip()):
            self.set_image(name)


    def _build(self, root):
        ctrl_frame = Tkinter.Frame(root)
        label = Tkinter.Label(ctrl_frame, text='Threshold')
        label.focus()
        self._threshold = Tkinter.Entry(ctrl_frame, width=8)
        self._threshold.insert(0, '0.5')
        btn = Tkinter.Button(ctrl_frame, text='Apply',
                command=self._pre_detector)
        root.bind('<Return>', self._pre_detector)

        label.pack(side='left')
        self._threshold.pack(side='left')
        btn.pack(side='left')
        ctrl_frame.pack(side='top')

        detector_frame = Tkinter.Frame(root)
        labelw = Tkinter.Label(detector_frame, text='wwidth')
        self._wwidth = Tkinter.Entry(detector_frame, width=3)
        self._wwidth.insert(0, '3')
        labelh = Tkinter.Label(detector_frame, text='wheight')
        self._wheight = Tkinter.Entry(detector_frame, width=3)
        self._wheight.insert(0, '3')
        labels = Tkinter.Label(detector_frame, text='sigma')
        self._sigma = Tkinter.Entry(detector_frame, width=3)
        self._sigma.insert(0, '2')
        self._invert = Tkinter.IntVar()
        invert = Tkinter.Checkbutton(detector_frame, text='Invert',
                variable=self._invert)

        labelw.pack(side='left')
        self._wwidth.pack(side='left')
        labelh.pack(side='left')
        self._wheight.pack(side='left')
        labels.pack(side='left')
        self._sigma.pack(side='left')
        invert.pack(side='left')
        detector_frame.pack(side='top')

        img_frame = Tkinter.Frame(root)
        #self._img_show = Tkinter.Label(img_frame)
        self._img_show = scrolledcanvas(root, img_frame, highlightthickness=0)

        self._img_show.pack(fill='both', expand=True)
        img_frame.pack(fill='both', expand=True)

        popup = Tkinter.Menu(root, tearoff=False)
        popup.add_command(label="Load", command=self._load_img)
        if sys.platform == 'darwin':
            root.bind('<2>', lambda evt: popup.post(evt.x_root, evt.y_root))
        root.bind('<Button-3>', lambda evt: popup.post(evt.x_root, evt.y_root))



root = Tkinter.Tk()
root.title('Harris Detector')
app = App(root)
app.set_image(sys.argv[1])
root.mainloop()
