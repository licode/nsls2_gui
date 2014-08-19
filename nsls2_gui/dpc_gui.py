from __future__ import print_function

import os
import sys
import multiprocessing as mp
import time

from PyQt4 import (QtCore, QtGui)
from PyQt4.QtCore import Qt
import matplotlib.cm as cm
import Image
import PIL
import scipy
from scipy.misc import imsave
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import ImageEnhance

from nsls2_gui import dpc_gui_kernel as dpc


sys.path.insert(0, '/home/nanopos/ecli/')
import pyspecfile

SOLVERS = ['Nelder-Mead',
           'Powell',
           'CG',
           'BFGS',
           'Newton-CG',
           'Anneal',
           'L-BFGS-B',
           'TNC',
           'COBYLA',
           'SLS   QP',
           'dogleg',
           'trust-ncg',
           ]

roi_x1 = 0
roi_x2 = 0
roi_y1 = 0
roi_y2 = 0
a = None
gx = None
gy = None
phi = None


CMAP_PREVIEW_PATH = os.path.join(os.path.dirname(__file__), '.cmap_previews')

def brush_to_color_tuple(brush):
    r, g, b, a = brush.color().getRgbF()
    return (r, g, b)


class DPCThread(QtCore.QThread):
    def __init__(self, canvas, pool=None, parent=None):
        QtCore.QThread.__init__(self, parent)
        DPCThread.instance = self
        self.canvas = canvas
        self.pool = pool
        self.fig = None

    def update_display(self, a, gx, gy, phi, flag=None): # ax is a pyplot object
        def show_image(ax, image):
            #return ax.imshow(np.flipud(image.T), interpolation='nearest',
            #                 origin='lower', cmap=cm.Greys_r)
            return ax.imshow(image, interpolation='nearest',
                             origin='lower', cmap=cm.Greys_r)
        
        def show_image_line(ax, image, start, end, direction=1):
            if direction == 1:
                ax.axhspan(start, end, facecolor='0.5', alpha=0.5)
                return ax.imshow(image, interpolation='nearest',
                                 origin='lower', cmap=cm.Greys_r)
            if direction == -1:
                ax.axvspan(start, end, facecolor='0.5', alpha=0.5)
                return ax.imshow(image, interpolation='nearest',
                                 origin='lower', cmap=cm.Greys_r)

        main = DPCDialog.instance
        canvas = self.canvas
        fig = canvas.figure
        fig.clear()
        fig.subplots_adjust(top=0.95, left=0, right=0.95, bottom=0)

        gs = gridspec.GridSpec(2, 2)

        if main.ion_data is not None:
            pixels = a.shape[0] * a.shape[1]
            ion_data = np.zeros(pixels)
            ion_data[:len(main.ion_data)] = main.ion_data
            ion_data[len(main.ion_data):] = ion_data[0]
            ion_data = ion_data.reshape(a.shape)

            min_ = np.min(a[np.where(a > 0)])
            a[np.where(a == 0)] = min_

            canvas.a_ax = a_ax = fig.add_subplot(gs[0, 1])
            a_ax.set_title('a')
            a_data = a / ion_data * ion_data[0]
            canvas.ima = ima = show_image(a_ax, a_data)
            fig.colorbar(ima)

        canvas.gx_ax = gx_ax = fig.add_subplot(gs[1, 0])
        gx_ax.set_title('X')
        canvas.imx = imx = show_image(gx_ax, gx)
        fig.colorbar(imx)

        canvas.gy_ax = gy_ax = fig.add_subplot(gs[1, 1])
        gy_ax.set_title('Y')
        canvas.imy = imy = show_image(gy_ax, gy)
        fig.colorbar(imy)
        
        """
        def onclick(event):
            print ('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, event.xdata, event.ydata))
        cid = gy_ax.canvas.mpl_connect('button_press_event', onclick)
        """
            
        if phi is not None:
            if main.ion_data is not None:
                phi_ax = fig.add_subplot(gs[0, 0])
            else:
                phi_ax = fig.add_subplot(gs[0, :])

            canvas.phi_ax = phi_ax

            phi_ax.set_title('phi')
            if flag == None:
                canvas.imphi = imphi = show_image(phi_ax, phi)
            if flag == "strap":
                canvas.imphi = imphi = show_image_line(phi_ax, phi, 
                                       DPCDialog.instance.strap_start.value(),
                                       DPCDialog.instance.strap_end.value(),
                                       DPCDialog.instance.direction)
            fig.colorbar(imphi)
            imphi.set_cmap(main._color_map)
            """
            def onclick(event):
                print ('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                event.button, event.x, event.y, event.xdata, event.ydata))
            cid = phi_ax.canvas.mpl_connect('button_press_event', onclick)
            """
        canvas.draw()

    def run(self):
        print('DPC thread started')
        try:
            ret = dpc.main(pool=self.pool, display_fcn=self.update_display,
                           **self.dpc_settings)
            print('DPC finished')
            global a
            global gx
            global gy
            global phi
            a, gx, gy, phi = ret

            main = DPCDialog.instance
            main.a, main.gx, main.gy, main.phi = a, gx, gy, phi
            self.update_display(a, gx, gy, phi)
            DPCDialog.instance.line_btn.setEnabled(True)
            #DPCDialog.instance.direction_btn.setEnabled(True)
            #DPCDialog.instance.removal_btn.setEnabled(True)
            #DPCDialog.instance.confirm_btn.setEnabled(True)
        finally:
            DPCDialog.instance.set_running(False)
    

class MplCanvas(FigureCanvas):
    """
    Canvas which allows us to use matplotlib with pyqt4
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        # We want the axes cleared every time plot() is called
        self.axes = fig.add_subplot(1, 1, 1)

        self.axes.hold(False)

        FigureCanvas.__init__(self, fig)

        # self.figure
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self._title = ''
        self.title_font = {'family': 'serif', 'fontsize': 10}
        self._title_size = 0
        self.figure.subplots_adjust(top=0.95, bottom=0.15)

        window_brush = self.window().palette().window()
        fig.set_facecolor(brush_to_color_tuple(window_brush))
        fig.set_edgecolor(brush_to_color_tuple(window_brush))
        self._active = False

    def _get_title(self):
        return self._title

    def _set_title(self, title):
        self._title = title
        if self.axes:
            self.axes.set_title(title, fontdict=self.title_font)
            #bbox = t.get_window_extent()
            #bbox = bbox.inverse_transformed(self.figure.transFigure)
            #self._title_size = bbox.height
            #self.figure.subplots_adjust(top=1.0 - self._title_size)

    title = property(_get_title, _set_title)
    
class Label(QtGui.QLabel):
    def __init__(self, parent = None):
        super(Label, self).__init__(parent)
        self.rubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)
        self.origin = QtCore.QPoint()
    
    def mousePressEvent(self, event):
        global roi_x1
        global roi_y1
        self.rubberBand.hide()
        if event.button() == Qt.LeftButton:
            self.origin = QtCore.QPoint(event.pos())
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubberBand.show()
            roi_x1 = event.pos().x()
            roi_y1 = event.pos().y()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.NoButton:
            pos = event.pos()
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())
            
    def mouseReleaseEvent(self, event):
        global roi_x2
        global roi_y2
        roi_x2 = event.pos().x()
        roi_y2 = event.pos().y()
        if((roi_x1, roi_y1)!=(roi_x2, roi_y2)):
            DPCDialog.instance.roi_x1_widget.setValue(roi_x1)
            DPCDialog.instance.roi_y1_widget.setValue(roi_y1)
            DPCDialog.instance.roi_x2_widget.setValue(roi_x2)
            DPCDialog.instance.roi_y2_widget.setValue(roi_y2)          
        else:
            if DPCDialog.instance.bad_flag != 0:
                DPCDialog.instance.bad_pixels_widget.addItem('%d, %d' % 
                (event.pos().x(), event.pos().y()))
                self.rubberBand.show()


class paintLabel(QtGui.QLabel):
    def __init__(self, parent = None):
        super(paintLabel, self).__init__(parent)
    
    def paintEvent(self, event):
        super(paintLabel, self).paintEvent(event)
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawLine(event, qp)
        qp.end()
         
    def drawLine(self, event, qp):
        size = self.size()
        pen = QtGui.QPen(QtCore.Qt.red)
        qp.setPen(pen)
        qp.drawLine(size.width()/2, 0, size.width()/2, size.height()-1)
        qp.drawLine(size.width()/2 - 1, 0, size.width()/2 - 1, size.height()-1)
        qp.drawLine(0, size.height()/2, size.width()-1, size.height()/2)
        qp.drawLine(0, size.height()/2-1, size.width()-1, size.height()/2-1)
        
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setColor(QtCore.Qt.black)
        qp.setPen(pen)
        qp.drawLine(0, 0, size.width()-1, 0)
        qp.drawLine(0, size.height()-1, size.width()-1, size.height()-1)
        qp.drawLine(0, 0, 0, size.height()-1)
        qp.drawLine(size.width()-1, 0, size.width()-1, size.height()-1)
        
   
        
class DPCDialog(QtGui.QDialog):
    CM_DEFAULT = 'jet'
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        DPCDialog.instance = self

        self.bin_num = 2**16
        self._thread = None
        self.ion_data = None
        self.bad_flag = 0
        self.direction = 1 # 1 for horizontal and -1 for vertical
        
        self.gx, self.gy, self.phi, self.a = None, None, None, None
        self.file_widget = QtGui.QLineEdit('Chromosome_9_%05d.tif')
        self.file_widget.setFixedWidth(400)
        self.focus_widget = QtGui.QDoubleSpinBox()

        self.dx_widget = QtGui.QDoubleSpinBox()
        self.dy_widget = QtGui.QDoubleSpinBox()
        self.pixel_widget = QtGui.QSpinBox()
        self.energy_widget = QtGui.QDoubleSpinBox()
        self.rows_widget = QtGui.QSpinBox()
        self.cols_widget = QtGui.QSpinBox()
        self.roi_x1_widget = QtGui.QSpinBox()
        self.roi_x2_widget = QtGui.QSpinBox()
        self.roi_y1_widget = QtGui.QSpinBox()
        self.roi_y2_widget = QtGui.QSpinBox()
        self.strap_start = QtGui.QSpinBox()
        self.strap_end = QtGui.QSpinBox()

        self.bad_pixels_widget = QtGui.QListWidget()
        self.bad_pixels_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bad_pixels_widget.customContextMenuRequested.connect(self._bad_pixels_menu)

        self.ref_widget = QtGui.QSpinBox()
        self.first_widget = QtGui.QSpinBox()

        self.processes_widget = QtGui.QSpinBox()
        self.solver_widget = QtGui.QComboBox()

        for solver in SOLVERS:
            self.solver_widget.addItem(solver)

        self.start_widget = QtGui.QPushButton('S&tart')
        self.stop_widget = QtGui.QPushButton('&Stop')
        self.save_widget = QtGui.QPushButton('Sa&ve')

        self.scan_button = QtGui.QPushButton('Load from s&can')
        
        self.color_map = QtGui.QComboBox()
        self.update_color_maps()
        self.color_map.currentIndexChanged.connect(self._set_color_map)
        self._color_map = mpl.cm.get_cmap(self.CM_DEFAULT)
        
        self.start_widget.clicked.connect(self.start)
        self.stop_widget.clicked.connect(self.stop)
        self.save_widget.clicked.connect(self.save)
        self.scan_button.clicked.connect(self.load_from_scan)
        
        self.layout1 = QtGui.QFormLayout()
        self.settings_widget1 = QtGui.QFrame()
        self.settings_widget1.setLayout(self.layout1)
        self.layout1.setRowWrapPolicy(self.layout1.WrapAllRows)
        self.layout1.addRow('&File format', self.file_widget)
        self.layout1.addRow('Phi color map', self.color_map)
        
        self.layout2 = QtGui.QFormLayout()
        self.settings_widget2 = QtGui.QFrame()
        self.settings_widget2.setLayout(self.layout2)
        self.layout2.addRow('&X step (um)', self.dx_widget)
        self.layout2.addRow('P&ixel size (um)', self.pixel_widget)
        self.layout2.addRow('&Rows', self.rows_widget)
        self.layout2.addRow('&Reference image', self.ref_widget)
        self.layout2.addRow('ROI X1', self.roi_x1_widget)
        self.layout2.addRow('ROI Y1', self.roi_y1_widget)
        
        self.layout3 = QtGui.QFormLayout()
        self.settings_widget3 = QtGui.QFrame()
        self.settings_widget3.setLayout(self.layout3) 
        self.layout3.addRow('&Y step (um)', self.dy_widget)
        self.layout3.addRow('Energy (keV)', self.energy_widget)
        self.layout3.addRow('&Columns', self.cols_widget)
        self.layout3.addRow('&First image', self.first_widget)
        self.layout3.addRow('ROI X2', self.roi_x2_widget)       
        self.layout3.addRow('ROI Y2', self.roi_y2_widget) 
        
        self.splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.splitter1.setLineWidth(1)
        self.splitter1.minimumSizeHint()
        self.splitter1.addWidget(self.settings_widget2)
        self.splitter1.addWidget(self.settings_widget3)
        
        ## The ROI image related components
        # roi_image is the image used to select the ROI and it is a QPixmap object
        # roi_img shows the same image but in a PIL image format
        # roi_image_temp is the temporary image used to show roi_image with an
        # enhanced contrast and etc.
        self.roi_image = QtGui.QPixmap(str(self.file_widget.text()) % self.ref_widget.value())
        self.roi_img = Image.open(str(self.file_widget.text()) % self.ref_widget.value())
        self.calHist()
        self.preContrast()
        self.roi_image_x = self.roi_image.size().width()
        self.roi_image_y = self.roi_image.size().height()
        self.img_lbl = Label(self)        
        self.img_lbl.setPixmap(self.roi_image)
        self.img_lbl.setFixedWidth(self.roi_image_x)
        self.img_lbl.setFixedHeight(self.roi_image_y)
        self.temp_lbl = paintLabel(self)
        self.temp_lbl.setFixedWidth(168)
        self.temp_lbl.setFixedHeight(168)
        
        self.txt_lbl = QtGui.QLabel(self)
        
        self.img_btn = QtGui.QPushButton('Select an image')
        self.img_btn.clicked.connect(self.load_an_image)
        self.his_btn = QtGui.QPushButton('Histgram equalization')
        self.his_btn.setCheckable(True)
        self.his_btn.clicked[bool].connect(self.histgramEqua)
        self.bri_btn = QtGui.QPushButton('Brightest pixels')
        self.bri_btn.clicked.connect(self.select_bri_pixels)
        self.bad_btn = QtGui.QPushButton('Select bad pixels') 
        self.bad_btn.setCheckable(True)
        self.bad_btn.clicked[bool].connect(self.bad_enable)
        
        self.sld = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sld.valueChanged[int].connect(self.change_contrast)
        
        self.line_btn = QtGui.QPushButton('Add/Change the strap')
        self.line_btn.setEnabled(False)
        self.line_btn.clicked.connect(self.add_strap)
        self.direction_btn = QtGui.QPushButton('Change the direction')
        self.direction_btn.clicked.connect(self.change_direction)
        self.direction_btn.setEnabled(False)
        self.removal_btn = QtGui.QPushButton('Remove the background')
        self.removal_btn.clicked.connect(self.remove_background)
        self.removal_btn.setEnabled(False)
        self.confirm_btn = QtGui.QPushButton('Confirm')
        self.confirm_btn.clicked.connect(self.confirm)
        self.confirm_btn.setEnabled(False)
        
        self.layout4 = QtGui.QFormLayout()
        self.settings_widget4 = QtGui.QFrame()
        self.settings_widget4.setLayout(self.layout4)
        self.layout4.addRow(self.txt_lbl)
        self.layout4.addRow(self.img_btn, self.his_btn)
        self.layout4.addRow(self.bri_btn, self.bad_btn)
        self.layout4.addRow(self.sld)
        self.layout4.addRow(self.temp_lbl)
        self.splitter3 = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter3.setLineWidth(0.1)        
        self.splitter3.minimumSizeHint()
        self.splitter3.addWidget(self.img_lbl)
        self.splitter3.addWidget(self.settings_widget4)
        
        self.layout5 = QtGui.QFormLayout()
        self.settings_widget5 = QtGui.QFrame()
        self.settings_widget5.setLayout(self.layout5)
        self.layout5.addRow('Bad pixels', self.bad_pixels_widget)
        self.layout5.addRow('F&ocus to detector (um)', self.focus_widget)
        self.layout5.addRow('&Solver method', self.solver_widget)
        self.layout5.addRow('&Processes', self.processes_widget)
        self.layout5.addRow('Strap start', self.strap_start)
        self.layout5.addRow('Strap end', self.strap_end)
        self.layout5.addRow(self.line_btn, self.direction_btn)
        self.layout5.addRow(self.removal_btn, self.confirm_btn)
        self.layout5.addRow(self.stop_widget, self.start_widget)
        self.layout5.addRow(self.save_widget, self.scan_button)
        
        self.splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter2.setLineWidth(1)
        self.splitter2.minimumSizeHint()
        self.splitter2.addWidget(self.settings_widget1)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.addWidget(self.settings_widget5)
        
        self.last_path = ''
        
        self._settings = {
            'file_format': [lambda: self.file_format, lambda value: self.file_widget.setText(value)],
            'dx': [lambda: self.dx, lambda value: self.dx_widget.setValue(float(value))],
            'dy': [lambda: self.dy, lambda value: self.dy_widget.setValue(float(value))],

            'x1': [lambda: self.roi_x1, lambda value: self.roi_x1_widget.setValue(int(value))],
            'y1': [lambda: self.roi_y1, lambda value: self.roi_y1_widget.setValue(int(value))],
            'x2': [lambda: self.roi_x2, lambda value: self.roi_x2_widget.setValue(int(value))],
            'y2': [lambda: self.roi_y2, lambda value: self.roi_y2_widget.setValue(int(value))],

            'pixel_size': [lambda: self.pixel_size, lambda value: self.pixel_widget.setValue(float(value))],
            'focus_to_det': [lambda: self.focus, lambda value: self.focus_widget.setValue(float(value))],
            'energy': [lambda: self.energy, lambda value: self.energy_widget.setValue(float(value))],

            'rows': [lambda: self.rows, lambda value: self.rows_widget.setValue(int(value))],
            'cols': [lambda: self.cols, lambda value: self.cols_widget.setValue(int(value))],
            'first_image': [lambda: self.first_image, lambda value: self.first_widget.setValue(int(value))],
            'ref_image': [lambda: self.ref_image, lambda value: self.ref_widget.setValue(int(value))],
            'processes': [lambda: self.processes, lambda value: self.processes_widget.setValue(int(value))],
            'bad_pixels': [lambda: self.bad_pixels, lambda value: self.set_bad_pixels(value)],
            'solver': [lambda: self.solver, lambda value: self.set_solver(value)],

            'last_path': [lambda: self.last_path, lambda value: setattr(self, 'last_path', value)],
            #'color_map': [lambda: self._color_map, lambda value: setattr(self, 'last_path', value)],
        }

        for w in [self.pixel_widget, self.focus_widget, self.energy_widget,
                  self.dx_widget, self.dy_widget, self.rows_widget, self.cols_widget,
                  self.roi_x1_widget, self.roi_x2_widget, self.roi_y1_widget, self.roi_y2_widget,
                  self.ref_widget, self.first_widget,
                  ]:
            w.setMinimum(0)
            w.setMaximum(int(2 ** 31 - 1))
            try:
                w.setDecimals(3)
            except:
                pass

        self.canvas = MplCanvas(width=8, height=0.25, dpi=50)

        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.setLineWidth(1)
        self.splitter.minimumSizeHint()

        self.splitter.addWidget(self.splitter3)
        self.splitter.addWidget(self.splitter2)
        self.splitter.addWidget(self.canvas)
        
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)
        
        self.load_settings()
    
    def add_strap(self, pressed):
        """
        Add two lines in the Phi image
        """
        self.confirm_btn.setEnabled(False)
        self.direction_btn.setEnabled(True)
        self.removal_btn.setEnabled(True)
        DPCThread.instance.update_display(a, gx, gy, phi, "strap")
    
    def change_direction(self, pressed):
        """
        Change the orientation of the strap
        """
        self.direction = -self.direction
        DPCThread.instance.update_display(a, gx, gy, phi, "strap")
        
    def remove_background(self, pressed):
        """
        Remove the background of the phase image
        """
        global phi
        self.confirm_btn.setEnabled(True)
        self.direction_btn.setEnabled(False)
        if self.direction == 1:
            strap = phi[self.strap_start.value():self.strap_end.value(), :]
            line = np.mean(strap, axis=0)
            self.phi_r = phi - line
            DPCThread.instance.update_display(a, gx, gy, self.phi_r)
            
        if self.direction == -1:
            strap = phi[:, self.strap_start.value():self.strap_end.value()]
            line = np.mean(strap, axis=1)
            self.phi_r = np.transpose(phi)
            self.phi_r = self.phi_r - line
            self.phi_r = np.transpose(self.phi_r)
            DPCThread.instance.update_display(a, gx, gy, self.phi_r)
    
    def confirm(self, pressed):
        """
        Confirm the background removal
        """
        global phi
        phi = self.phi_r
        imsave('phi.jpg', phi)
        np.savetxt('phi.txt', phi)
        self.confirm_btn.setEnabled(False)
        self.direction_btn.setEnabled(False)
        self.removal_btn.setEnabled(False)
        
    def bad_enable(self, pressed):
        """
        Enable or disable bad pixels selection by changing the bad_flag value
        """
        if pressed:
            self.bad_flag = 1
        else:
            self.bad_flag = 0
        
    def histgramEqua(self, pressed):
        """
        Histogram equalization for the ROI image
        """
        if pressed:
            self.roi_image_temp = QtGui.QPixmap('equalizedImg.tif')
            self.img_lbl.setPixmap(self.roi_image_temp)
        else:
            self.img_lbl.setPixmap(self.roi_image)
    
    def preContrast(self):
        self.contrastImage = self.roi_img.convert('L')
        self.enh = ImageEnhance.Contrast(self.contrastImage)
            
    def calHist(self):
        """
        Calculate the histogram of the image used to select ROI
        """
        img = np.array(self.roi_img.getdata(), dtype=np.uint16)
        imhist,bins = np.histogram(img, bins=self.bin_num, range=(0, self.bin_num), density=True)
        cdf = imhist.cumsum()
        cdf = (self.bin_num-1) * cdf / cdf[-1]
        equalizedImg = np.uint16(np.floor(np.interp(img, bins[:-1], cdf)))
        equalizedImg = np.reshape(equalizedImg, (self.roi_img.size[1], self.roi_img.size[0]), order='C')
        scipy.misc.imsave('equalizedImg.tif', equalizedImg)
    
    def select_bri_pixels(self):
        """
        Select the bad pixels (pixels with the maximum pixel value)
        """
        img = np.array(self.roi_img.getdata(), dtype=np.uint16)
        array = np.reshape(img, (self.roi_img.size[1], self.roi_img.size[0]), order='C')
        indices = np.where(array==array.max())
        indices_num = indices[0].size
        for i in range(indices_num):
            self.bad_pixels_widget.addItem('%d, %d' % (indices[1][i], indices[0][i]))
               
    def change_contrast(self, value):
        """
        Change the contrast of the ROI image by slider bar
        """
        delta = value / 10.0
        self.enh.enhance(delta).save('change_contrast.tif')
        contrastImageTemp = QtGui.QPixmap('change_contrast.tif')
        self.img_lbl.setPixmap(contrastImageTemp)
        
    def eventFilter(self, source, event):
        """
        Event filter to enable cursor coordinates tracking on the ROI image
        """
        if (event.type() == QtCore.QEvent.MouseMove and 
            source is self.img_lbl):
            if event.buttons() == QtCore.Qt.NoButton:
                pos = event.pos()
                self.txt_lbl.setText('x=%d, y=%d, value=%d ' % (pos.x(), 
                                     pos.y(), self.roi_img.getpixel((pos.x(), pos.y()))))
                            
                top_left_x = pos.x()-10 if pos.x()-10>=0 else 0
                top_left_y = pos.y()-10 if pos.y()-10>=0 else 0
                bottom_right_x = pos.x()+10 if pos.x()+10<self.roi_img.size[0] else self.roi_img.size[0]-1
                bottom_right_y = pos.y()+10 if pos.y()+10<self.roi_img.size[1] else self.roi_img.size[1]-1
                
                if (pos.y()-10)<0:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignBottom)
                if (pos.x()+10)>=self.roi_img.size[0]:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignLeft)
                if (pos.x()-10)<0:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignRight)
                if (pos.y()+10)>=self.roi_img.size[1]:
                    self.temp_lbl.setAlignment(QtCore.Qt.AlignTop)
                
                width = bottom_right_x - top_left_x + 1
                height = bottom_right_y - top_left_y+ 1
                img_fraction = self.img_lbl.pixmap().copy(top_left_x, top_left_y, width, height)
                scaled_img_fraction = img_fraction.scaled(width*8, height*8)
                self.temp_lbl.setPixmap(scaled_img_fraction)
                
        if (event.type() == QtCore.QEvent.MouseMove and 
            source is not self.img_lbl):
            if event.buttons() == QtCore.Qt.NoButton:
                self.txt_lbl.setText('')
                self.temp_lbl.clear()
            
        return QtGui.QDialog.eventFilter(self, source, event)
        
    def load_an_image(self):
        """
        Load an image to select the ROI
        """
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
                '/home')
        self.roi_temp_image = QtGui.QPixmap(fname)
        if not self.roi_temp_image.isNull():
            self.roi_image = self.roi_temp_image
            self.roi_img = Image.open(str(fname))
            self.calHist()
            self.preContrast()
            self.roi_image_x = self.roi_image.size().width()
            self.roi_image_y = self.roi_image.size().height()
            self.img_lbl.setPixmap(self.roi_image)
            self.img_lbl.setFixedWidth(self.roi_image_x)
            self.img_lbl.setFixedHeight(self.roi_image_y)
        
    def _set_color_map(self, index):
        """
        User changed color map callback.
        """
        cm_ = str(self.color_map.itemText(index))
        print('Color map set to: %s' % cm_)
        self._color_map = mpl.cm.get_cmap(cm_)
        try:
            for im in [self.canvas.imphi, ]:
                im.set_cmap(self._color_map)
        except Exception as ex:
            print('failed to set color map: (%s) %s' % (ex.__class__.__name__, 
                                                        ex))
        finally:
            self.canvas.draw()

    def create_cmap_previews(self):
        """
        Create the color map previews for the combobox
        """
        cm_names = sorted(_cm for _cm in mpl.cm.datad.keys()
                          if not _cm.endswith('_r'))
        cm_filenames = [os.path.join(CMAP_PREVIEW_PATH, '%s.png' % cm_name)
                        for cm_name in cm_names]

        ret = zip(cm_names, cm_filenames)
        points = np.outer(np.ones(10), np.arange(0, 1, 0.01))
        if not os.path.exists(CMAP_PREVIEW_PATH):
            try:
                os.mkdir(CMAP_PREVIEW_PATH)
            except Exception as ex:
                print('Unable to create preview path: %s' % ex)

            return ret

        for cm_name, fn in zip(cm_names, cm_filenames):
            if not os.path.exists(fn):
                print('Generating colormap preview: %s' % fn)
                canvas = MplCanvas(width=2, height=0.25, dpi=50)
                fig = canvas.figure
                fig.clear()

                ax = fig.add_subplot(1, 1, 1)
                ax.axis("off")
                fig.subplots_adjust(top=1, left=0, right=1, bottom=0)
                _cm = mpl.cm.get_cmap(cm_name)
                ax.imshow(points, aspect='auto', cmap=_cm, origin='lower')
                try:
                    fig.savefig(fn)
                except Exception as ex:
                    print('Unable to create color map preview "%s"' % fn,
                          file=sys.stderr)
                    break

        return ret

    def update_color_maps(self):
        size = None
        for i, (cm_name, fn) in enumerate(self.create_cmap_previews()):
            print('Color map', fn)
            if os.path.exists(fn):
                self.color_map.addItem(QtGui.QIcon(fn), cm_name)
                if size is None:
                    size = QtGui.QPixmap(fn).size()
                    self.color_map.setIconSize(size)
            else:
                self.color_map.addItem(cm_name)

            if cm_name == self.CM_DEFAULT:
                self.color_map.setCurrentIndex(i)

    @property
    def settings(self):
        return QtCore.QSettings('BNL', 'DPC-GUI')

    def save_settings(self):
        settings = self.settings
        for key, (getter, setter) in self._settings.items():
            settings.setValue(key, getter())

        settings.setValue('geometry', self.geometry())
        settings.setValue('splitter', self.splitter.saveState())

    def load_settings(self):
        settings = self.settings
        for key, (getter, setter) in self._settings.items():
            value = settings.value(key).toPyObject()
            if value is not None:
                setter(value)

        try:
            self.setGeometry(settings.value('geometry').toPyObject())
            self.splitter.restoreState(settings.value('splitter').toByteArray())
        except:
            pass
    
    def closeEvent(self, event=None):
        self.save_settings()

    @property
    def dx(self):
        return float(self.dx_widget.text())

    @property
    def dy(self):
        return float(self.dy_widget.text())

    @property
    def processes(self):
        return int(self.processes_widget.text())

    @property
    def file_format(self):
        return str(self.file_widget.text())

    @property
    def pixel_size(self):
        return self.pixel_widget.value()

    @property
    def focus(self):
        return self.focus_widget.value()

    @property
    def energy(self):
        return self.energy_widget.value()

    @property
    def rows(self):
        return self.rows_widget.value()

    @property
    def cols(self):
        return self.cols_widget.value()

    @property
    def first_image(self):
        return self.first_widget.value()

    @property
    def ref_image(self):
        return self.ref_widget.value()

    @property
    def roi_x1(self):
        return self.roi_x1_widget.value()

    @property
    def roi_x2(self):
        return self.roi_x2_widget.value()

    @property
    def roi_y1(self):
        return self.roi_y1_widget.value()

    @property
    def roi_y2(self):
        return self.roi_y2_widget.value()

    @property
    def bad_pixels(self):
        pixels = []
        w = self.bad_pixels_widget

        def fix_tuple(item):
            item = str(item.text())
            return [int(x) for x in item.split(',')]

        return [fix_tuple(w.item(i)) for i in range(w.count())]

    def _bad_pixels_menu(self, pos):
        def add():
            s, ok = QtGui.QInputDialog.getText(self, 'Position?', 'Position in the format: x, y')
            if ok:
                s = str(s)
                x, y = s.split(',')
                x = int(x)
                y = int(y)
                self.bad_pixels_widget.addItem('%d, %d' % (x, y))

        def remove():
            rows = [index.row() for index in self.bad_pixels_widget.selectedIndexes()]
            for row in reversed(sorted(rows)):
                self.bad_pixels_widget.takeItem(row)
        
        def clear():
            self.bad_pixels_widget.clear()

        self.menu = menu = QtGui.QMenu()
        add_action = menu.addAction('&Add', add)
        remove_action = menu.addAction('&Remove', remove)
        clear_action = menu.addAction('&Clear', clear)

        menu.popup(self.bad_pixels_widget.mapToGlobal(pos))

    def load_from_scan(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Scan filename', self.last_path, '*.spec')
        if not filename:
            return

        self.last_path = filename

        print('Loading %s' % filename)
        with pyspecfile.SPECFileReader(filename, parse_data=False) as f:
            scans = dict((int(scan['number']), scan) for scan in f.scans)
            scan_info = ['%04d - %s' % (number, scan['command'])
                         for number, scan in scans.items()
                         if 'mesh' in scan['command']]

            scan_info.sort()
            print('\n'.join(scan_info))

            s, ok = QtGui.QInputDialog.getItem(self, 'Scan selection', 'Scan number?', scan_info, 0, False)
            if ok:
                print('Selected scan', s)
                number = int(s.split(' ')[0])
                sd = scans[number]
                f.parse_data(sd)

                timepix_index = sd['columns'].index('tpx_image')
                line0 = sd['lines'][0]
                timepix_first_image = int(line0[timepix_index])

                try:
                    ion1_index = sd['columns'].index('Ion1')
                    self.ion_data = np.array([line[ion1_index] for line in sd['lines']])
                except Exception as ex:
                    print('Failed loading Ion1 data (%s) %s' % (ex, ex.__class__.__name__))
                    self.ion_data = None

                print('First timepix image:', timepix_first_image)

                self.ref_widget.setValue(timepix_first_image - 1)
                self.first_widget.setValue(timepix_first_image - 1)

                command = sd['command'].replace('  ', ' ')

                x = [2, 3, 4]  # x start, end, points
                y = [6, 7, 8]  # y start, end, points
                info = command.split(' ')

                x_info = [float(info[i]) for i in x]
                y_info = [float(info[i]) for i in y]

                dx = (x_info[1] - x_info[0]) / (x_info[2] - 1)
                dy = (y_info[1] - y_info[0]) / (y_info[2] - 1)

                self.rows_widget.setValue(int(y_info[-1]))
                self.cols_widget.setValue(int(x_info[-1]))

                self.dx_widget.setValue(float(dx))
                self.dy_widget.setValue(float(dy))

    @property
    def solver(self):
        return SOLVERS[self.solver_widget.currentIndex()]

    def set_solver(self, solver):
        self.solver_widget.setCurrentIndex(SOLVERS.index(solver))

    def set_bad_pixels(self, pixels):
        w = self.bad_pixels_widget
        w.clear()
        for item in pixels:
            x, y = item
            w.addItem('%d, %d' % (x, y, ))

    @property
    def dpc_settings(self):
        ret = {}
        for key, (getter, setter) in self._settings.items():
            ret[key] = getter()
        return ret

    def start(self):
        self.line_btn.setEnabled(False)
        self.direction_btn.setEnabled(False)
        self.removal_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        
        if self._thread is not None and self._thread.isFinished():
            self._thread = None

        if self._thread is None:
            if self.processes == 0:
                pool = None
            else:
                pool = mp.Pool(processes=self.processes)

            thread = self._thread = DPCThread(self.canvas, pool=pool)
            thread.dpc_settings = self.dpc_settings
            del thread.dpc_settings['processes']
            del thread.dpc_settings['last_path']
            thread.start()
            self.set_running(True)

    def set_running(self, running):
        self.start_widget.setEnabled(not running)
        self.stop_widget.setEnabled(running)

    def stop(self):
        if self._thread is not None:
            pool = self._thread.pool
            if pool is not None:
                pool.terminate()
                self._thread.pool = None

            time.sleep(0.2)
            self._thread.terminate()
            self._thread = None
            self.set_running(False)

    def save(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save filename prefix', '', '')
        if not filename:
            return

        arrays = [('gx', self.gx),
                  ('gy', self.gy),
                  ('phi', self.phi),
                  ('a', self.a)]

        for name, arr in arrays:
            im = PIL.Image.fromarray(arr)
            im.sasve('%s_%s.tif' % (filename, name))
            np.savetxt('%s_%s.txt' % (filename, name), im)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    dialog = DPCDialog()
    dialog.show()
    app.installEventFilter(dialog)

    
    sys.exit(app.exec_())
