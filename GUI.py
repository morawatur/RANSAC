# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ransac.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup

import numpy as np
from PyQt4 import QtCore, QtGui

# -------------------------------------------------------------------

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

# -------------------------------------------------------------------

class Ui_MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi()

    def setupUi(self):
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(750, 580)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        self.img_view = QtGui.QLabel(self.centralwidget)
        # self.img_view = QtGui.QGraphicsView(self.centralwidget)
        self.img_view.setGeometry(QtCore.QRect(20, 10, 512, 512))
        self.img_view.setObjectName(_fromUtf8("img_view"))

        imagePath = QtGui.QFileDialog.getOpenFileName()
        self.image = dm3.ReadDm3File(imagePath)
        self.createPixmap()

        self.verticalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(560, 20, 160, 281))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.n_iter_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.n_iter_label.setEnabled(True)
        self.n_iter_label.setObjectName(_fromUtf8("n_iter_label"))
        self.verticalLayout.addWidget(self.n_iter_label)
        self.n_iter_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.n_iter_edit.setMinimum(1)
        self.n_iter_edit.setMaximum(2000)
        self.n_iter_edit.setSingleStep(10)
        self.n_iter_edit.setProperty("value", 100)
        self.n_iter_edit.setObjectName(_fromUtf8("n_iter_edit"))
        self.verticalLayout.addWidget(self.n_iter_edit)
        self.n_inl_threshold_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.n_inl_threshold_label.setObjectName(_fromUtf8("n_inl_threshold_label"))
        self.verticalLayout.addWidget(self.n_inl_threshold_label)
        self.n_inl_threshold_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.n_inl_threshold_edit.setMinimum(1000)
        self.n_inl_threshold_edit.setMaximum(20000)
        self.n_inl_threshold_edit.setSingleStep(100)
        self.n_inl_threshold_edit.setProperty("value", 7000)
        self.n_inl_threshold_edit.setObjectName(_fromUtf8("n_inl_threshold_edit"))
        self.verticalLayout.addWidget(self.n_inl_threshold_edit)
        self.try_again_threshold_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.try_again_threshold_label.setObjectName(_fromUtf8("try_again_threshold_label"))
        self.verticalLayout.addWidget(self.try_again_threshold_label)
        self.try_again_threshold_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.try_again_threshold_edit.setMinimum(100)
        self.try_again_threshold_edit.setMaximum(50000)
        self.try_again_threshold_edit.setSingleStep(100)
        self.try_again_threshold_edit.setProperty("value", 3000)
        self.try_again_threshold_edit.setObjectName(_fromUtf8("try_again_threshold_edit"))
        self.verticalLayout.addWidget(self.try_again_threshold_edit)
        self.min_dist_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.min_dist_label.setObjectName(_fromUtf8("min_dist_label"))
        self.verticalLayout.addWidget(self.min_dist_label)
        self.min_dist_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.min_dist_edit.setMinimum(1)
        self.min_dist_edit.setMaximum(100)
        self.min_dist_edit.setProperty("value", 5)
        self.min_dist_edit.setObjectName(_fromUtf8("min_dist_edit"))
        self.verticalLayout.addWidget(self.min_dist_edit)
        self.max_n_tries_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.max_n_tries_label.setObjectName(_fromUtf8("max_n_tries_label"))
        self.verticalLayout.addWidget(self.max_n_tries_label)
        self.n_tries_edit = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.n_tries_edit.setMinimum(1)
        self.n_tries_edit.setMaximum(100)
        self.n_tries_edit.setProperty("value", 20)
        self.n_tries_edit.setObjectName(_fromUtf8("n_tries_edit"))
        self.verticalLayout.addWidget(self.n_tries_edit)
        self.min_ab_ratio_label = QtGui.QLabel(self.verticalLayoutWidget)
        self.min_ab_ratio_label.setObjectName(_fromUtf8("min_ab_ratio_label"))
        self.verticalLayout.addWidget(self.min_ab_ratio_label)
        self.min_ab_ratio_edit = QtGui.QDoubleSpinBox(self.verticalLayoutWidget)
        self.min_ab_ratio_edit.setMaximum(1.0)
        self.min_ab_ratio_edit.setSingleStep(0.05)
        self.min_ab_ratio_edit.setProperty("value", 0.3)
        self.min_ab_ratio_edit.setObjectName(_fromUtf8("min_ab_ratio_edit"))
        self.verticalLayout.addWidget(self.min_ab_ratio_edit)
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(560, 330, 160, 181))
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.find_model_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.find_model_button.setObjectName(_fromUtf8("find_model_button"))
        self.verticalLayout_2.addWidget(self.find_model_button)
        self.export_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.export_button.setObjectName(_fromUtf8("export_button"))
        self.verticalLayout_2.addWidget(self.export_button)
        self.crop_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.crop_button.setObjectName(_fromUtf8("crop_button"))
        self.verticalLayout_2.addWidget(self.crop_button)
        self.img_view.raise_()
        self.verticalLayoutWidget.raise_()
        self.n_iter_label.raise_()
        self.verticalLayoutWidget_2.raise_()
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 750, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "RANSAC Ellipse Fitting", None))
        self.n_iter_label.setText(_translate("MainWindow", "No of iterations", None))
        self.n_inl_threshold_label.setText(_translate("MainWindow", "Num. of inliers (threshold)", None))
        self.try_again_threshold_label.setText(_translate("MainWindow", "Try-again threshold", None))
        self.min_dist_label.setText(_translate("MainWindow", "Min. dist. from model [px]", None))
        self.max_n_tries_label.setText(_translate("MainWindow", "Max. num. of tries", None))
        self.min_ab_ratio_label.setText(_translate("MainWindow", "Min. a-b ratio", None))
        self.find_model_button.setText(_translate("MainWindow", "Start", None))
        self.export_button.setText(_translate("MainWindow", "Export", None))
        self.crop_button.setText(_translate("MainWindow", "Crop", None))

    def createPixmap(self):
        # paddedImage = imsup.PadImageBufferToNx512(self.image, np.max(self.image.buffer))
        # qImg = QtGui.QImage(imsup.ScaleImage(paddedImage.buffer, 0.0, 255.0).astype(np.uint8),
        #                     paddedImage.width, paddedImage.height, QtGui.QImage.Format_Indexed8)
        qImg = QtGui.QImage(imsup.ScaleImage(self.image, 0.0, 255.0).astype(np.uint8),
                            self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap(qImg)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)  # !!!
        self.img_view.setPixmap(pixmap)

# -------------------------------------------------------------------

def run_ransac_window():
    import sys
    app = QtGui.QApplication(sys.argv)
    ransac_win = Ui_MainWindow()
    sys.exit(app.exec_())

run_ransac_window()