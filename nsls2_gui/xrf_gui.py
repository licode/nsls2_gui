
import numpy as np
import six
import sys
import os
import json
from collections import OrderedDict

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt

from nsls2.fitting.model.xrf_model import (ModelSpectrum, set_range, k_line,
                                           l_line, m_line, get_linear_model, PreFitAnalysis)
from nsls2.fitting.model.background import snip_method
from nsls2.constants import Element


def change_factory(hi_box, low_box):
    @QtCore.Slot(int)
    def inner(bound_type_index):
        if bound_type_index == 0 or bound_type_index == 1:
            hi_box.setDisabled(True)
            low_box.setDisabled(True)
        else:
            hi_box.setDisabled(False)
            low_box.setDisabled(False)
    return inner


def update_value(valuebox):
    @QtCore.Slot(unicode)
    def inner(value):
        valuebox.setSingleStep(float(value))
    return inner


def update_param_name(name):
    if name == 'e_linear':
        name = 'energy calibration linear'
    elif name == 'e_offset':
        name = 'energy calibration constant'
    elif name == 'e_quadratic':
        name = 'energy calibration quadratic'
    elif name == 'coherent_sct_amplitude':
        name = 'elastic peak amplitude'
    elif name == 'coherent_sct_energy':
        name = 'incident beam energy'
    return name.replace('_', ' ')


class XRFFit(QtGui.QMainWindow):

    def __init__(self):
        super(XRFFit, self).__init__()
        self.initUI()

    def initUI(self):

        # experiment widget
        import_data = self.import_data_widget()

        #self.pw = PlotWindow(incident_energy=self.pre_dictv['coherent_sct_energy']['value'])
        self.pw = PlotWindow(incident_energy=11)

        # plot data
        plot_tab = self.plot_widget()

        # set parameter
        set_param = self.parameter_widget()

        # 1D fit widget
        myfit = QtGui.QWidget()
        fit_layout = QtGui.QVBoxLayout()
        myfit.setLayout(fit_layout)

        # 2D fit widget
        myfit2D = QtGui.QWidget()
        fit2D_layout = QtGui.QVBoxLayout()
        myfit2D.setLayout(fit2D_layout)

        #self.addTab(import_data, 'Import')
        #self.addTab(plot_tab, 'Find Element')
        #self.addTab(set_param, 'Set Parameter')
        #self.addTab(myfit, 'Fit')
        #self.addTab(myfit2D, '2D MAP')

        #openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        #openFile.setShortcut('Ctrl+O')
        #openFile.setStatusTip('Open new File')
        #openFile.triggered.connect(self.showDialog)

        self.dockWdg1 = QtGui.QDockWidget(self)
        #mw.content1 = QtGui.QWidget()
        self.dockWdg1.setWidget(import_data)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWdg1)
        self.dockWdg1.setWindowTitle("Import Data")

        self.dockWdg2 = QtGui.QDockWidget(self)
        self.dockWdg2.setWidget(plot_tab)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWdg2)
        self.dockWdg2.setWindowTitle("Find Element")

        self.dockWdg3 = QtGui.QDockWidget(self)
        #self.pw.setGeometry(300, 300, 800, 700)
        self.dockWdg3.setWidget(set_param)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWdg3)
        self.dockWdg3.setWindowTitle("Set Parameter")

        self.dockWdg4 = QtGui.QDockWidget(self)
        #self.pw.setGeometry(300, 300, 800, 700)
        self.dockWdg4.setWidget(self.pw)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWdg4)
        self.dockWdg4.setWindowTitle("Plot")

        #self.setGeometry(300, 300, 800, 700)
        self.setWindowTitle('XRF Fit')
        #self.raise_()
        #self.show()
        #self.activateWindow()

    def import_data_widget(self):
        """Import folder and data files."""

        exp = QtGui.QWidget()
        self.exp_layout = QtGui.QVBoxLayout()

        valbox = QtGui.QLabel("<font size=8> PyXRF: X-ray Fluorescence Analysis Tool </font>")

        # open experiment file
        folder_w = QtGui.QWidget()
        folder_l = QtGui.QHBoxLayout()

        folder_n = QtGui.QLabel('Choose Folder')
        folder_b = QtGui.QPushButton('Open')
        folder_b.clicked.connect(self.open_folder)

        folder_l.addWidget(folder_n)
        folder_l.addWidget(folder_b)
        folder_w.setLayout(folder_l)

        # open single file
        file_w = QtGui.QWidget()
        file_l = QtGui.QHBoxLayout()
        file_n = QtGui.QLabel('Choose Single File')

        file_check = QtGui.QCheckBox('Choose Single File')
        file_check.stateChanged.connect(self.change_element_plot)

        file_b = QtGui.QPushButton('Open Single File')
        file_b.clicked.connect(self.open_experiment_data)

        file_l.addWidget(file_n)
        #file_l.addWidget(file_check)
        file_l.addWidget(file_b)
        #file_l.addStretch(1)
        file_w.setLayout(file_l)

        # open multiple files
        files_w = QtGui.QWidget()
        files_l = QtGui.QHBoxLayout()
        file_ns = QtGui.QLabel('Choose Multiple Files')

        files_check = QtGui.QCheckBox('Choose Multiple Files')
        files_check.stateChanged.connect(self.change_element_plot)

        #files_l.addStretch(file_ns)
        files_l.addWidget(files_check)
        #files_l.addStretch(1)
        files_w.setLayout(files_l)

        btn2 = QtGui.QPushButton('Open multiple experiment data', self)

        #self.exp_layout.addStretch(1)
        self.exp_layout.addWidget(valbox)
        #self.exp_layout.addStretch(1)
        self.exp_layout.addWidget(folder_w)
        self.exp_layout.addWidget(file_w)
        self.exp_layout.addWidget(files_w)
        self.exp_layout.addWidget(btn2)

        #self.exp_layout.addStretch(1)

        # folder path
        self.status_w1 = QtGui.QWidget()
        self.status_l1 = QtGui.QGridLayout()
        self.status_w1.setLayout(self.status_l1)

        # file name
        self.status_w2 = QtGui.QWidget()
        self.status_l2 = QtGui.QGridLayout()
        self.status_w2.setLayout(self.status_l2)

        self.exp_layout.addWidget(self.status_w1)
        self.exp_layout.addWidget(self.status_w2)

        exp.setLayout(self.exp_layout)
        return exp

    def plot_widget(self,
                    filepath='/Users/Li/Research/X-ray/Research_work/all_code/PyXRF/test/xrf_parameter.json'):
        """Plot 1D data with option as linear or log plot."""

        #open default configuration file
        json_data = open(filepath, 'r')
        self.param_dictv = json.load(json_data)
        self.pre_dictv = self.param_dictv.copy()

        element_tab = QtGui.QWidget()
        #plot_tab = QtGui.QMainWindow()
        element_layout = QtGui.QVBoxLayout()

        #incident energy can be obtained from instrument file
        self.pw = PlotWindow(incident_energy=self.pre_dictv['coherent_sct_energy']['value'])

        # linear or log
        hwidget = QtGui.QWidget()
        hlayout = QtGui.QHBoxLayout()
        plot_choice = QtGui.QComboBox()
        inputlist = ['log', 'linear']
        plot_choice.addItems(inputlist)
        plot_choice.setCurrentIndex(0)
        plot_choice.currentIndexChanged.connect(self.pw.define_option)

        hlayout.addWidget(plot_choice)

        # plot button
        #plot_w = QtGui.QWidget()
        #plot_l = QtGui.QHBoxLayout()
        plot_btn = QtGui.QPushButton('Plot Raw Spectrum', self)
        plot_btn.clicked.connect(self.plot_experiment_data)
        #plot_l.addWidget(plot_btn)
        #plot_l.addStretch(1)
        #plot_w.setLayout(plot_l)
        hlayout.addWidget(plot_btn)

        hlayout.addStretch(1)
        hwidget.setLayout(hlayout)
        element_layout.addWidget(hwidget)

        #element_layout.addWidget(plot_w)

        self.pre_total = {}

        intro_l = QtGui.QLabel('Automatic Peak Finding')
        intro_l.setToolTip('Use non-negative least squares method to search all possible elements. User needs to define global\n'
                           'parameters related to peak position and peak width.')
        element_layout.addWidget(intro_l)

        param_w = QtGui.QWidget()
        e_grid = QtGui.QGridLayout()

        fixed_s = 200

        f1_n = QtGui.QLabel('fwhm fanoprime')
        f1_v = QtGui.QLineEdit(str(self.pre_dictv['fwhm_fanoprime']['value']))
        f1_v.setFixedWidth(fixed_s)
        f1_s = QtGui.QLabel('(suggested: 0.0)')
        self.pre_total.update(fwhm_fanoprime=f1_v)

        f2_n = QtGui.QLabel('fwhm offset')
        f2_v = QtGui.QLineEdit(str(self.pre_dictv['fwhm_offset']['value']))
        f2_v.setFixedWidth(fixed_s)
        f2_s = QtGui.QLabel('(suggested: 0.178)')
        self.pre_total.update(fwhm_offset=f2_v)

        ecal1_n = QtGui.QLabel('energy calibration constant')
        ecal1_v = QtGui.QLineEdit(str(self.pre_dictv['e_offset']['value']))
        ecal1_v.setFixedWidth(fixed_s)
        ecal1_s = QtGui.QLabel('(suggested: 0.007)')
        self.pre_total.update(e_offset=ecal1_v)

        ecal2_n = QtGui.QLabel('energy calibration linear')
        ecal2_v = QtGui.QLineEdit(str(self.pre_dictv['e_linear']['value']))
        ecal2_v.setFixedWidth(fixed_s)
        ecal2_s = QtGui.QLabel('(suggested: 0.01)')
        self.pre_total.update(e_linear=ecal2_v)

        ecal3_n = QtGui.QLabel('energy calibration quadratic')
        ecal3_v = QtGui.QLineEdit(str(self.pre_dictv['e_quadratic']['value']))
        ecal3_v.setFixedWidth(fixed_s)
        ecal3_s = QtGui.QLabel('(suggested: 0.0)')
        self.pre_total.update(e_quadratic=ecal3_v)

        fcomp_n = QtGui.QLabel('compton fwhm correction')
        fcomp_v = QtGui.QLineEdit(str(self.pre_dictv['compton_fwhm_corr']['value']))
        fcomp_v.setFixedWidth(fixed_s)
        fcomp_s = QtGui.QLabel('(suggested: 1.5)')
        self.pre_total.update(compton_fwhm_corr=fcomp_v)

        non_fit = {}
        elow_n = QtGui.QLabel('energy range low [keV]')
        elow_v = QtGui.QLineEdit(str(self.pre_dictv['non_fitting_values']['energy_bound_low']))
        elow_v.setFixedWidth(fixed_s)
        elow_s = QtGui.QLabel('(suggested: 0.0)')
        non_fit.update(energy_bound_low=elow_v)

        ehigh_n = QtGui.QLabel('energy range high [keV]')
        ehigh_v = QtGui.QLineEdit(str(self.pre_dictv['non_fitting_values']['energy_bound_high']))
        ehigh_v.setFixedWidth(fixed_s)
        ehigh_s = QtGui.QLabel('(suggested: 12.0)')
        non_fit.update(energy_bound_high=ehigh_v)

        self.pre_total.update(non_fitting_values=non_fit)

        e_grid.addWidget(f1_n, 0, 0)
        e_grid.addWidget(f1_v, 0, 1)
        e_grid.addWidget(f1_s, 0, 2)
        e_grid.addWidget(f2_n, 1, 0)
        e_grid.addWidget(f2_v, 1, 1)
        e_grid.addWidget(f2_s, 1, 2)
        e_grid.addWidget(fcomp_n, 2, 0)
        e_grid.addWidget(fcomp_v, 2, 1)
        e_grid.addWidget(fcomp_s, 2, 2)
        e_grid.addWidget(ecal1_n, 3, 0)
        e_grid.addWidget(ecal1_v, 3, 1)
        e_grid.addWidget(ecal1_s, 3, 2)
        e_grid.addWidget(ecal2_n, 4, 0)
        e_grid.addWidget(ecal2_v, 4, 1)
        e_grid.addWidget(ecal2_s, 4, 2)
        e_grid.addWidget(ecal3_n, 5, 0)
        e_grid.addWidget(ecal3_v, 5, 1)
        e_grid.addWidget(ecal3_s, 5, 2)

        e_grid.addWidget(elow_n, 6, 0)
        e_grid.addWidget(elow_v, 6, 1)
        e_grid.addWidget(elow_s, 6, 2)
        e_grid.addWidget(ehigh_n, 7, 0)
        e_grid.addWidget(ehigh_v, 7, 1)
        e_grid.addWidget(ehigh_s, 7, 2)

        param_w.setLayout(e_grid)
        element_layout.addWidget(param_w)

        # run button
        r_w = QtGui.QWidget()
        r_box = QtGui.QHBoxLayout()
        r_btn = QtGui.QPushButton('Run')
        r_btn.clicked.connect(self.save_parameter_pre)
        r_btn.clicked.connect(self.pre_fit)
        #e_grid.addWidget(r_btn, 8, 0)
        r_box.addWidget(r_btn)

        save_btn = QtGui.QPushButton('Save')
        #e_grid.addWidget(save_btn, 8, 1)
        r_box.addWidget(save_btn)

        saveas_btn = QtGui.QPushButton('Save As')
        #e_grid.addWidget(saveas_btn, 8, 2)
        r_box.addWidget(saveas_btn)
        r_w.setLayout(r_box)
        element_layout.addWidget(r_w)

        # output elements from auto fit
        rec_w = QtGui.QWidget()
        rec_box = QtGui.QHBoxLayout()
        result_n = QtGui.QLabel('Recommended Elements')
        result_v = QtGui.QLineEdit()
        rec_box.addWidget(result_n)
        rec_box.addWidget(result_v)
        rec_w.setLayout(rec_box)
        #element_layout.addWidget(rec_w)

        # manual peak finding
        m_label = QtGui.QLabel('Manual Peak Finding')
        element_layout.addWidget(m_label)

        # element plot
        hwidget_e = QtGui.QWidget()
        hlayout_e = QtGui.QHBoxLayout()

        element_check = QtGui.QCheckBox('Show Element')
        element_check.stateChanged.connect(self.change_element_plot)

        hlayout_e.addWidget(element_check)
        hlayout_e.addStretch(1)
        hwidget_e.setLayout(hlayout_e)

        hwidget_scroll = QtGui.QWidget()
        hlayout_scroll = QtGui.QHBoxLayout()

        self.element_choice = QtGui.QComboBox()
        inputlist_e = ['Choose element by using scroll arrow.'] + k_line + l_line + m_line
        self.element_choice.addItems(inputlist_e)
        self.element_choice.setCurrentIndex(0)
        self.element_choice.currentIndexChanged.connect(self.plot_experiment_data_element)
        self.element_choice.setEnabled(False)
        hlayout_scroll.addWidget(self.element_choice)
        hlayout_scroll.addStretch(1)
        hwidget_scroll.setLayout(hlayout_scroll)

        #hlayout_e.addWidget(self.element_choice)
        #hlayout_e.addStretch(1)
        element_layout.addWidget(hwidget_e)
        element_layout.addWidget(hwidget_scroll)

        # final elements to be used
        final_n = QtGui.QLabel('Elements to be used')
        final_v = QtGui.QLineEdit()
        element_layout.addWidget(final_n)
        element_layout.addWidget(final_v)

        element_layout.addStretch(1)
        element_tab.setLayout(element_layout)
        return element_tab

    def pre_fit(self):
        """prefit to perform initial guess of elements"""
        y = np.loadtxt(str(self.fileObj))
        self.prefit_x, self.total_y, sorted_result = pre_fit_linear(self.pre_dictv, y)
        self.plot_experiment_data_prefit()
        #w1 = MessageWindow()

        w1 = QtGui.QDialog(self)
        #A = QtGui.QDockWidget(w1)
        w1.setWindowTitle('Automatic fitting results')
        t1 = QtGui.QLabel('Elements to be used')
        h_layout = QtGui.QVBoxLayout()
        h_layout.addWidget(t1)

        g_layout = QtGui.QGridLayout()
        for i in range(len(sorted_result)):
            v = sorted_result[i]
            element_v0 = QtGui.QLabel(str(v[0]))
            element_v1 = QtGui.QLabel(str(v[1]))
            g_layout.addWidget(element_v0, i, 0)
            g_layout.addWidget(element_v1, i, 1)
        w1.setLayout(g_layout)
        #w1.setGeometry(300, 300, 50, 50)
        w1.show()

    def pre_fit_output(self):
        res_dict = OrderedDict(zip(total_list, np.sum(total_y, axis=0)*0.01))

        #new_dict = {k: v for k,v in res_dict.iteritems() if v!=0}
        sorted_list = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
        for k, v in res_dict.items():
            if v != 0:
                print (k, v)

        for data in sorted_list:
            if data[1] != 0:
                print(data[0], data[1])

    def change_element_plot(self, state):
        """Plot element or not."""
        if state == QtCore.Qt.Checked:
            self.element_choice.setEnabled(True)
        else:
            self.element_choice.setEnabled(False)

    def parameter_widget(self):

        # parameter widget
        param = QtGui.QWidget()
        p_layout = QtGui.QVBoxLayout()

        # open configuration file
        horz_widget = QtGui.QWidget()
        horz_layout = QtGui.QHBoxLayout()

        btn = QtGui.QPushButton('Open configuration file', self)
        btn.clicked.connect(self.open_configuration_file)

        #p_layout.addStretch(1)
        horz_layout.addWidget(btn)
        horz_layout.addStretch(1)
        horz_widget.setLayout(horz_layout)
        p_layout.addWidget(horz_widget)

        self.newone = QtGui.QWidget()
        p_layout.addWidget(self.newone)

        self.grid = QtGui.QGridLayout()

        param.setLayout(p_layout)
        self.param = param
        return param

    def open_folder(self):
        """Define folder path."""
        dir_v = '.'
        self.folderObj = QtGui.QFileDialog.getExistingDirectory(self, 'Open File Dialog',
                                                                directory=dir_v)
        folderbox = QtGui.QLabel("Folder name: {0}".format(self.folderObj))

        # clean up first
        while self.status_l1.count():
            item = self.status_l1.takeAt(0)
            item.widget().deleteLater()
        self.status_l1.addWidget(folderbox, 0, 0)

    def open_experiment_data(self):
        """Load single experiment data"""
        dir_v = self.folderObj
        self.fileObj = QtGui.QFileDialog.getOpenFileName(self, 'Open File Dialog',
                                                         directory=dir_v)

        pathlist = str(self.fileObj).split('/')
        filebox = QtGui.QLabel("File name: {0}".format(pathlist[-1]))

        # clean up first
        while self.status_l2.count():
            item = self.status_l2.takeAt(0)
            item.widget().deleteLater()
        self.status_l2.addWidget(filebox, 0, 0)

    def plot_experiment_data(self):

        # get x,y data from file, then assign them to global variables
        self.y0 = np.loadtxt(str(self.fileObj))
        self.x0 = np.arange(len(self.y0))

        # clean up prefit data
        self.pw.total_y = []

        self.pw.plot(self.x0*0.01, self.y0)
        self.pw.show()

    def plot_experiment_data_prefit(self):

        # change range based on dict data
        self.x0, self.y0 = set_range(self.pre_dictv, self.x0, self.y0)

        self.pw.set_prefit_data(self.prefit_x, self.total_y)
        self.pw.plot(self.x0*0.01, self.y0)
        self.pw.show()

    def plot_experiment_data_element(self, v):

        if v == 0:
            return
        else:
            v = v-1

        #if main.exec_():
        self.pw.set_element(v)
        self.pw.plot(self.x0*0.01, self.y0)
        self.pw.show()

    def open_configuration_file(self):
        """configuration file saves all the fitting parameters"""

        dir_v = '.'

        fileObj = QtGui.QFileDialog.getOpenFileName(self, 'Open File Dialog',
                                                    directory=dir_v, filter='Json files (*.json)')

        # open json data
        json_data = open(fileObj, 'r')
        self.dictv = json.load(json_data)

        while self.grid.count():
            item = self.grid.takeAt(0)
            item.widget().deleteLater()
        self.set_parameter()

    def set_parameter(self):
        """
        Show all parameters in grid based on given dictionary.
        """
        grid = self.grid

        save_btn = QtGui.QPushButton('Save', self)
        grid.addWidget(save_btn, 0, 1)
        save_btn.clicked.connect(self.save_parameter)

        saveas_btn = QtGui.QPushButton('Save As', self)
        grid.addWidget(saveas_btn, 0, 2)
        saveas_btn.clicked.connect(self.save_as_parameter)

        # update button
        #update_btn = QtGui.QPushButton('Update plot', self)
        #grid.addWidget(update_btn, 0, 3)
        #update_btn.clicked.connect(self.update_plot)

        # run button
        cal_btn = QtGui.QPushButton('Forward calculation', self)
        grid.addWidget(cal_btn, 0, 3)
        cal_btn.clicked.connect(self.guess_and_plot)

        # fit button
        fit_btn = QtGui.QPushButton('Initial Fit', self)
        grid.addWidget(fit_btn, 0, 4)
        fit_btn.clicked.connect(self.fit_and_plot)
        #self.connect(run_btn, QtCore.SIGNAL('clicked()'), self.update_plot)

        #self.connect(self, QtCore.SIGNAL('FitDone()'), self.fit_done)

        dictv = self.dictv

        self.total_box = {}

        vlist = ['Name', 'Value', 'Step', 'Bound', 'Low', 'High']

        pos = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]

        non_fit_dict = {}

        element_label = QtGui.QLabel('Define element list')
        element_edit = QtGui.QLineEdit(dictv['non_fitting_values']['element_list'])
        grid.addWidget(element_label, pos[0][0], pos[0][1])
        grid.addWidget(element_edit, pos[1][0], pos[1][1], 1, -1)
        non_fit_dict.update(element_list=element_edit)

        e_range_label = QtGui.QLabel('Define energy range')
        grid.addWidget(e_range_label, pos[0][0]+1, pos[0][1])

        low_e_label = QtGui.QLabel('Low')
        low_e_edit = QtGui.QDoubleSpinBox()
        low_e_edit.setValue(dictv['non_fitting_values']['energy_bound_low'])
        low_e_edit.setRange(0, 100)
        low_e_edit.setSingleStep(0.1)
        low_e_edit.valueChanged.connect(self.save_parameter)
        low_e_edit.valueChanged.connect(self.update_plot)

        #low_e_edit = QtGui.QLineEdit(str(dictv['non_fitting_values']['energy_bound_low']))
        grid.addWidget(low_e_label, pos[1][0]+1, pos[1][1])
        grid.addWidget(low_e_edit, pos[2][0]+1, pos[2][1])

        non_fit_dict.update(energy_bound_low=low_e_edit)

        high_e_edit = QtGui.QDoubleSpinBox()
        high_e_edit.setValue(dictv['non_fitting_values']['energy_bound_high'])
        high_e_edit.setRange(0, 1000)
        high_e_edit.setSingleStep(0.1)
        high_e_edit.valueChanged.connect(self.save_parameter)
        high_e_edit.valueChanged.connect(self.update_plot)

        high_e_label = QtGui.QLabel('High')
        #high_e_edit = QtGui.QLineEdit(str(dictv['non_fitting_values']['energy_bound_high']))
        grid.addWidget(high_e_label, pos[3][0]+1, pos[3][1])
        grid.addWidget(high_e_edit, pos[4][0]+1, pos[4][1])
        high_e_edit.valueChanged.connect(self.save_parameter)
        high_e_edit.valueChanged.connect(self.update_plot)

        non_fit_dict.update(energy_bound_high=high_e_edit)
        self.total_box.update(non_fitting_values=non_fit_dict)

        start_index = 3
        for i in range(len(vlist)):
            #valbox = QtGui.QLabel("<font size=12>"+vlist[i] + "</font>")
            valbox = QtGui.QLabel(vlist[i])
            grid.addWidget(valbox, pos[i][0]+start_index, pos[i][1])

        dictv = OrderedDict(sorted(six.iteritems(dictv)))

        for i, v in enumerate(six.iteritems(dictv)):

            if v[0] == 'non_fitting_values':
                continue

            if 'ka1' in v[0] or 'ka2' in v[0] or 'kb1' in v[0] or 'kb2' in v[0]:
                if 'ka1_area' not in v[0]:
                    continue

            vnew = update_param_name(v[0])

            namebox = QtGui.QLabel(str(vnew))
            #valuebox = QtGui.QLineEdit(str(v[1]['value']))
            valuebox = QtGui.QDoubleSpinBox()
            valuebox.setRange(-1e9, 1e9)
            valuebox.setDecimals(4)
            valuebox.setValue(np.float(v[1]['value']))

            value_maintainer = update_value(valuebox)

            valuebox.valueChanged.connect(self.guess_and_plot)

            set_step = QtGui.QLineEdit(str(1.0))
            set_step.textChanged.connect(value_maintainer)

            btypebox = QtGui.QComboBox()
            inputlist = ['fixed', 'none', 'lohi']
            btypebox.addItems(inputlist)
            indexv = inputlist.index(str(v[1]['bound_type']))
            btypebox.setCurrentIndex(indexv)

            minv = QtGui.QLineEdit(str(v[1]['min']))
            maxv = QtGui.QLineEdit(str(v[1]['max']))

            if indexv == 0 or indexv == 1:
                minv.setDisabled(True)
                maxv.setDisabled(True)
            else:
                minv.setDisabled(False)
                maxv.setDisabled(False)

            box_state_maintainer = change_factory(minv, maxv)

            btypebox.currentIndexChanged.connect(box_state_maintainer)

            namelist = [namebox, valuebox, set_step,
                        btypebox, minv, maxv]

            for j in range(len(namelist)):
                grid.addWidget(namelist[j], pos[j][0]+i+start_index+1, pos[j][1])

            #data_box.append(namelist[j])
            self.total_box.update({v[0]: {'value': valuebox, 'bound_type': btypebox,
                                          'min': minv, 'max': maxv}})

        #self.set_parameter(grid)

        #self.param.setLayout(grid)
        self.newone.setLayout(grid)

        self.newone.repaint()

    def save_parameter_pre(self):
        """Save paramters from prefit"""

        for k, v in six.iteritems(self.pre_total):
            if k == 'non_fitting_values':
                #itemdict = {'element_list': str(v['element_list'].text()),
                #            'energy_bound_low': float(v['energy_bound_low'].text()),
                #            'energy_bound_high': float(v['energy_bound_high'].text())}
                for k1, v1 in six.iteritems(self.pre_total[k]):
                    self.pre_dictv[k][str(k1)] = float(v1.text())
            else:
                #itemdict = {'value': float(v['value'].text()), 'bound_type': str(v['bound_type'].currentText()),
                #            'min': float(v['min'].text()), 'max': float(v['max'].text())}
                self.pre_dictv[str(k)]['value'] = float(v.text())

    def save_parameter(self):

        #dir = '.'

        #fileObj = QtGui.QFileDialog.getSaveFileName(self, 'Save File Dialog',
        #                                            directory=dir,
        #                                            filter='Json files (*.json)')
        self.current_dict = {}
        for k, v in six.iteritems(self.total_box):
            if k == 'non_fitting_values':
                itemdict = {'element_list': str(v['element_list'].text()),
                            'energy_bound_low': float(v['energy_bound_low'].text()),
                            'energy_bound_high': float(v['energy_bound_high'].text())}
            else:
                itemdict = {'value': float(v['value'].text()), 'bound_type': str(v['bound_type'].currentText()),
                            'min': float(v['min'].text()), 'max': float(v['max'].text())}

            self.current_dict.update({k: itemdict})

    def save_as_parameter(self):

        dir_v = '.'

        fileObj = QtGui.QFileDialog.getSaveFileName(self, 'Save As File Dialog',
                                                    directory=dir_v,
                                                    filter='Json files (*.json)')

        self.current_dict = {}
        for k, v in six.iteritems(self.total_box):
                if k == 'non_fitting_values':
                    itemdict = {'element_list': str(v['element_list'].text()),
                                'energy_bound_low': float(v['energy_bound_low'].text()),
                                'energy_bound_high': float(v['energy_bound_high'].text())}
                else:
                    itemdict = {'value': float(v['value'].text()), 'bound_type': str(v['bound_type'].currentText()),
                                'min': float(v['min'].text()), 'max': float(v['max'].text())}

                self.current_dict.update({k: itemdict})

        try:
            json_data = open(fileObj, 'w')
            json.dump(self.current_dict, json_data)
        except:
            pass
        #print current_dict
        #return current_dict

    def update_plot(self):

        if len(self.current_dict) != 0:
            x0, y0 = set_range(self.current_dict, self.x0, self.y0)
            # update global variables
            self.x0_fit = x0
            self.y0_fit = y0
        else:
            x0 = self.x0
            y0 = self.y0
        #x0 *= 0.01
        #main = Window(x0, y0, x0, result.best_fit+bg)
        self.pw.plot(x0*0.01, y0)
        self.pw.show()

    def guess_and_plot(self):

        self.save_parameter()

        try:
            x0 = self.x0_fit
            y0 = self.y0_fit
        except AttributeError, e:
            x0 = np.array(self.x0)
            y0 = np.array(self.y0)

        bg = snip_method(y0, 0, 0.01, 0)

        MS = ModelSpectrum(self.current_dict)
        p = MS.mod.make_params()
        #y_init = MS.mod.eval(x=x0, params=p)
        y_init = MS.mod.components[0].eval(x=x0, params=p) + MS.mod.components[1].eval(x=x0, params=p)
        #result, bg = DoFit(self.dictv, x0, y0)
        #print result.fit_report()

        #self.emit(QtCore.SIGNAL('FitDone()'))

        #main = Window(x0, y0, x0, result.best_fit+bg)
        #x0 = result.values['e_offset'] + result.values['e_linear'] * x0 +\
        #     result.values['e_quadratic'] * x0**2

        #self.pw.plot(x0*0.01, y0, x0*0.01, result.best_fit+bg)
        self.pw.plot(x0*0.01, y0, x0*0.01, y_init + bg)
        self.pw.show()

    def fit_and_plot(self):

        self.save_parameter()

        try:
            x0 = self.x0_fit
            y0 = self.y0_fit
        except AttributeError, e:
            x0 = np.array(self.x0)
            y0 = np.array(self.y0)

        #bg = snip_method(y0, 0, 0.01, 0)

        #MS = ModelSpectrum(self.dictv)
        #p = MS.mod.make_params()
        #y_init = MS.mod.components[0].eval(x=x0, params=p) + \
        #         MS.mod.components[1].eval(x=x0, params=p)
        result, bg = DoFit(self.current_dict, x0, y0)
        print result.fit_report()

        #self.emit(QtCore.SIGNAL('FitDone()'))

        #main = Window(x0, y0, x0, result.best_fit+bg)
        x0 = result.values['e_offset'] + result.values['e_linear'] * x0 +\
             result.values['e_quadratic'] * x0**2

        self.pw.plot(x0, y0, x0, result.best_fit+bg)
        self.pw.show()


class MessageWindow(QtGui.QDialog):

    def __init__(self, parent=None):
        super(MessageWindow, self).__init__(parent)
        t1 = QtGui.QLabel('Elements to be used')
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(t1)
        self.setLayout(self.layout)


class PlotWindow(QtGui.QDialog):

    def __init__(self, incident_energy,
                 parent=None):
        super(PlotWindow, self).__init__(parent)

        self.incident_energy = incident_energy
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        self.option = 'log'
        self.normv = False

        self.element_plot = False
        self.elist = []

        # for prefitted data
        self.total_y = []

        # set the layout
        self.layout = QtGui.QVBoxLayout()

        #self.layout.addWidget(self.toolbar)
        #self.layout.addWidget(self.canvas)
        #layout.addWidget(self.button)
        #self.setLayout(self.layout)

    def define_option(self, v):
        if v == 0:
            self.option = 'log'
        elif v == 1:
            self.option = 'linear'
        else:
            self.option = 'both'

    def set_element(self, v):

        total_list = k_line + l_line + m_line
        ename = total_list[v]

        if len(ename) <= 2:
            e = Element(ename)
            if e.cs(self.incident_energy)['ka1'] != 0:
                for i in range(4):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(self.incident_energy).all[i][1]/e.cs(self.incident_energy).all[0][1]))

        elif '_L' in ename:
            e = Element(ename[:-2])
            print e.cs(self.incident_energy)['la1']
            if e.cs(self.incident_energy)['la1'] != 0:
                for i in range(4, 17):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(self.incident_energy).all[i][1]/e.cs(self.incident_energy).all[4][1]))

        else:
            e = Element(ename[:-2])
            if e.cs(self.incident_energy)['ma1'] != 0:
                for i in range(17, 21):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(self.incident_energy).all[i][1]/e.cs(self.incident_energy).all[17][1]))

    def set_prefit_data(self, prefit_x, total_y):
        self.prefit_x = prefit_x
        self.total_y = total_y

    def plot(self, x, y, x1=None, y1=None, minv=1e-5):
        """ plot experiment results """

        while self.layout.count():
            item = self.layout.takeAt(0)
            item.widget().deleteLater()
        #self.layout = QtGui.QVBoxLayout()

        ax1 = self.figure.add_subplot(1, 1, 1)

        if self.option == 'linear':

            ax1.plot(x, y, 'o', markersize=2, label='experiment')

            if len(self.elist) != 0:
                ax1.hold(True)
                for i in range(len(self.elist)):
                    ax1.hold(True)
                    ax1.plot([self.elist[i][0], self.elist[i][0]],
                             [minv, self.elist[i][1]*np.max(y)], 'r-', linewidth=2.0)
                # need to clear the list every time
                self.elist = []

            if x1 is not None:
                ax1.hold(True)
                ax1.plot(x1, y1, '-')

            if len(self.total_y) > 0:
                ax1.hold(True)
                ax1.plot(self.prefit_x, np.sum(self.total_y, axis=1), 'b-', label='prefit')
                ax1.plot(self.prefit_x, self.total_y, 'g-')

            ax1.set_xlabel('Energy [keV]')
            ax1.set_ylabel('Counts')
            ax1.hold(False)

        elif self.option == 'log':
            ax1.semilogy(x, y, 'o', markersize=2)

            if len(self.elist) != 0:
                ax1.hold(True)
                for i in range(len(self.elist)):
                    ax1.semilogy([self.elist[i][0], self.elist[i][0]],
                                 [minv, self.elist[i][1]*np.max(y)], 'r-', linewidth=2.0)
                # need to clear the list every time
                self.elist = []

            if x1:
                ax1.hold(True)
                ax1.semilogy(x1, y1, '-')

            if len(self.total_y) > 0:
                ax1.hold(True)
                ax1.semilogy(self.prefit_x, np.sum(self.total_y, axis=1), 'b-', label='prefit')
                ax1.semilogy(self.prefit_x, self.total_y, 'g-')

            ax1.set_ylim([max(y)*minv, max(y)*2.0])
            ax1.set_xlabel('Energy [keV]')
            ax1.set_ylabel('Counts')
            ax1.hold(False)

        else:
            ax1 = self.figure.add_subplot(2, 1, 1)
            ax1.plot(x, y, 'o', markersize=2)

            if x1:
                ax1.hold(True)
                ax1.plot(x1, y1, '-')

            ax1.set_ylabel('Counts')
            ax1.hold(False)

            ax2 = self.figure.add_subplot(2, 1, 2)
            ax2.semilogy(x, y, 'o', markersize=2)

            if x1:
                ax2.hold(True)
                ax2.semilogy(x1, y1, '-')

            ax2.set_ylim([max(y)*1e-5, max(y)*2.0])
            ax2.set_xlabel('Energy [keV]')
            ax2.set_ylabel('Counts')
            ax2.hold(False)

        self.canvas.draw()
        some_button = QtGui.QPushButton('a button')
        #some_button.clicked.connect(self.canvas.draw)

        some_other_button = QtGui.QPushButton('another button!')

        #layout = QtGui.QVBoxLayout()

        layout_h = QtGui.QHBoxLayout()
        layout_h.addWidget(some_button)
        layout_h.addWidget(some_other_button)

        widget_h = QtGui.QWidget()
        widget_h.setLayout(layout_h)

        #self.layout.addWidget(widget_h)

        layout_v = QtGui.QVBoxLayout()
        layout_v.addWidget(self.toolbar)
        layout_v.addWidget(self.canvas)

        widget_v = QtGui.QWidget()

        widget_v.setLayout(layout_v)
        self.layout.addWidget(widget_v)

        self.toolbar.show()

        self.setLayout(self.layout)


class OpenFile(QtGui.QMainWindow):

    def __init__(self):
        super(OpenFile, self).__init__()
        self.initUI()

    def initUI(self):
        self.textEdit = QtGui.QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        #menubar = QtGui.QMainWindow.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('File dialog')
        self.show()

    def showDialog(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                '/home')

        f = open(fname, 'r')

        data = f.read()
        self.textEdit.setText(data)


def pre_fit_linear(parameter_dict, y0):
    """
    Run prefit to get initial elements.

    """

    # read json file
    x0 = np.arange(len(y0))

    x, y = set_range(parameter_dict, x0, y0)

    # get background
    bg = snip_method(y, 0, 0.01, 0)

    y = y - bg

    element_list = k_line + l_line
    new_element = ', '.join(element_list)
    parameter_dict['non_fitting_values']['element_list'] = new_element

    total_list = element_list + ['compton', 'elastic']
    total_list = [str(v) for v in total_list]

    matv = get_linear_model(x, parameter_dict)

    x = parameter_dict['e_offset']['value'] + parameter_dict['e_linear']['value']*x + \
        parameter_dict['e_quadratic']['value'] * x**2

    PF = PreFitAnalysis(y, matv)
    out, res = PF.nnls_fit_weight()
    total_y = out * matv

    result_dict = OrderedDict(zip(total_list, np.sum(total_y, axis=0)*0.01))

    sorted_result = sorted(six.iteritems(result_dict), key=lambda x: x[1], reverse=True)
    sorted_v = [v for v in sorted_result if v[1] != 0]

    for data in sorted_v:
        print(data[0], data[1])
    return x, total_y, sorted_v


def DoFit(parameter_dict, x, y):

    bg = snip_method(y, 0, 0.01, 0)

    c_val = 1e-2

    # first fit
    MS = ModelSpectrum(parameter_dict)
    result = MS.model_fit(x, y-bg, w=1/np.sqrt(y), maxfev=100,
                          xtol=c_val, ftol=c_val, gtol=c_val)
    #fitname = list(result1.values.keys())
    return result, bg


def main():

    # read json file
    config_file = 'xrf_parameter.json'
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             config_file)

    json_data = open(file_path, 'r')
    parameter_dict = json.load(json_data)

    app = QtGui.QApplication(sys.argv)
    test = XRFFit()

    #main = Window()
    #main.show()

    sys.exit(app.exec_())


def main1():

    app = QtGui.QApplication(sys.argv)
    ex = OpenFile()
    #main()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()
