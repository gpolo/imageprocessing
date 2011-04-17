# -*- encoding: utf8 -*-
import os
import sys
import numpy
import Tkinter
import tkFileDialog
from PIL import Image, ImageTk

#import matplotlib
#matplotlib.use('TkAgg') # XXX Esse backend deixa mais lento o plot,
#                        # mas se não usar não da pra mudar de imagem :/
#
#from matplotlib import pyplot
#from matplotlib.colors import BoundaryNorm

from harris import harris


def withdraw(event):
    widget = root.nametowidget(event.widget)
    while not hasattr(widget, 'withdraw'):
        widget = widget.master
    widget.withdraw()

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
        if isinstance(imgpath, basestring):
            src_img = Image.open(imgpath)
        else:
            src_img = imgpath
        root.image = src_img
        img = ImageTk.PhotoImage(src_img)
        canvas_addimage(self._img_show, img, *src_img.size)
        self._img_show.s_img = img
        self._img_show.img = src_img

    def _run_detector(self, event=None):
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

        images = harris(self._img_show.img, threshold, sigma, wwidth, wheight,
                invert=invert_img)
        mask, new_img = images[0], images[1]

        if new_img is None:
            return
        new_img_tk = ImageTk.PhotoImage(new_img)
        mimg = Image.new('L', (len(mask), len(mask[0])))
        mimg.putdata([255 * mask[i][j]
            for i in xrange(len(mask)) for j in xrange(len(mask[0]))])
        mimg = ImageTk.PhotoImage(mimg)

        window = Tkinter.Toplevel()
        window.image = new_img
        label = Tkinter.Label(window, text='%s, %s' % (threshold, sigma))
        mask_img = Tkinter.Label(window, image=mimg)
        mask_img.img = mimg
        img_frame = Tkinter.Frame(window)
        img = scrolledcanvas(window, img_frame, highlightthickness=0)
        img.new_img = new_img_tk
        canvas_addimage(img, new_img_tk,new_img_tk.width(),new_img_tk.height())
        label.pack()
        mask_img.pack()
        img.pack(fill='both', expand=True)
        img_frame.pack(fill='both', expand=True)

        for image in images[2:]:
            win = Tkinter.Toplevel()
            win.image = image
            iimg = ImageTk.PhotoImage(win.image)
            iframe = Tkinter.Frame(win)
            ilbl = scrolledcanvas(win, iframe, highlightthickness=0)
            canvas_addimage(ilbl, iimg, *image.size)
            ilbl.img = iimg
            ilbl.pack(fill='both', expand=True)
            iframe.pack(fill='both', expand=True)


    def _save(self):
        widget = root.nametowidget(root.focus_displayof())
        while not hasattr(widget, 'withdraw'):
            widget = widget.master
        try:
            image = widget.image
        except AttributeError:
            return
        else:
            self._save_img(image)

    def _save_img(self, image):
        name = tkFileDialog.asksaveasfilename(parent=root,
                initialdir=os.getcwd())
        if name is not None and len(name.strip()):
            image.save(name)

    def _load_img(self):
        name = tkFileDialog.askopenfilename(initialdir=os.getcwd())
        if name is not None and len(name.strip()):
            self.set_image(name)


    def _build(self, root):
        menu = Tkinter.Menu(name='apple')
        file_menu = Tkinter.Menu(menu)
        file_menu.add_command(label="Load", command=self._load_img)
        file_menu.add_command(label="Save", command=self._save)
        menu.add_cascade(label="Harris", menu=file_menu)
        root.event_add('<<Close>>', '<Command-w>')
        root.bind('<<Close>>', lambda evt: root.quit())
        root.bind_class('all', '<<Close>>', withdraw)
        root.config(menu=menu)

        ctrl_frame = Tkinter.Frame(root)
        label = Tkinter.Label(ctrl_frame, text='Threshold')
        label.focus()
        self._threshold = Tkinter.Entry(ctrl_frame, width=8)
        self._threshold.insert(0, '0.5')
        btn = Tkinter.Button(ctrl_frame, text='Apply',
                command=self._run_detector)
        root.bind('<Return>', self._run_detector)


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
        self._img_show = scrolledcanvas(root, img_frame, highlightthickness=0)

        self._img_show.pack(fill='both', expand=True)
        img_frame.pack(fill='both', expand=True)

        popup = Tkinter.Menu(root, tearoff=False)
        popup.add_command(label="Load", command=self._load_img)
        if sys.platform == 'darwin':
            root.bind('<2>', lambda evt: popup.post(evt.x_root, evt.y_root))
        root.bind('<Button-3>', lambda evt: popup.post(evt.x_root, evt.y_root))


if __name__ == "__main__":
    root = Tkinter.Tk()
    root.title('Harris Detector')
    app = App(root)
    if len(sys.argv) != 2:
        z = numpy.zeros((512, 512))
        z[30:40, 30:40] = numpy.ones((10, 10)) * 255
        x = numpy.fft.fftshift(numpy.fft.fft2(z))
        img = Image.fromarray(numpy.log(1 + abs(x))**2)
        app.set_image(img.convert('RGB'))
    else:
        app.set_image(sys.argv[1])
    root.mainloop()
