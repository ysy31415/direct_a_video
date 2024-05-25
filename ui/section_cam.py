from PyQt5.QtWidgets import (QApplication, QMainWindow, QCheckBox, QLabel, QVBoxLayout,QGroupBox,  QHBoxLayout, QPushButton, QGraphicsView, QSlider, QFileDialog,
                             QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsPathItem,
                             QLineEdit)
from PyQt5.QtCore import Qt

class CamSection(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.enableSwitch = QCheckBox("Enable Camera Movement Control")
        self.enableSwitch.setChecked(False)  # 默认设置为勾选状态
        self.enableSwitch.toggled.connect(self.toggleAvailability)  # 连接信号槽

        self.slider_cam_x = QSlider(Qt.Horizontal)
        self.slider_cam_x.setValue(0) # default
        self.slider_cam_x.setMinimum(-100)
        self.slider_cam_x.setMaximum(100)
        self.slider_cam_x.setMinimumWidth(200)
        self.value_cam_x = QLabel(f'{0:.2f}')
        self.slider_cam_x.valueChanged.connect(lambda: self.updateSliderLabel(self.slider_cam_x, self.value_cam_x))


        self.slider_cam_y = QSlider(Qt.Horizontal)
        self.slider_cam_y.setValue(0) # default
        self.slider_cam_y.setMinimum(-100)
        self.slider_cam_y.setMaximum(100)
        self.slider_cam_y.setMinimumWidth(200)
        self.value_cam_y = QLabel(f'{0:.2f}')
        self.slider_cam_y.valueChanged.connect(lambda: self.updateSliderLabel(self.slider_cam_y, self.value_cam_y))


        self.slider_cam_z = QSlider(Qt.Horizontal)
        self.slider_cam_z.setValue(100) # default
        self.slider_cam_z.setMinimum(-100)
        self.slider_cam_z.setMaximum(100)
        self.slider_cam_z.setMinimumWidth(200)
        self.value_cam_z = QLabel(f'{1:.2f}')
        self.slider_cam_z.valueChanged.connect(lambda: self.updateSliderLabelZ(self.slider_cam_z, self.value_cam_z))

        # cam cfg
        self.camCfgCheckbox = QCheckBox("Enable Camera CFG")
        self.camCfgCheckbox.setChecked(True)  # 默认设置为勾选状态

        # cam_off_t
        self.slider_cam_off = QSlider(Qt.Horizontal)
        self.slider_cam_off.setValue(85) # default
        self.slider_cam_off.setMinimum(0)
        self.slider_cam_off.setMaximum(100)
        self.slider_cam_off.setMinimumWidth(200)
        self.value_cam_off = QLabel(f'{0.85:.2f}')
        self.slider_cam_off.valueChanged.connect(lambda: self.updateSliderLabel(self.slider_cam_off, self.value_cam_off))

        # 重置按钮
        self.resetCamButton = QPushButton('Reset')
        self.resetCamButton.clicked.connect(self.reset)

        self.toggleAvailability(self.enableSwitch.isChecked())

    def toggleAvailability(self, checked):
        self.reset()
        self.slider_cam_x.setEnabled(checked)
        self.slider_cam_y.setEnabled(checked)
        self.slider_cam_z.setEnabled(checked)
        self.camCfgCheckbox.setEnabled(checked)
        self.slider_cam_off.setEnabled(checked)
        self.resetCamButton.setEnabled(checked)


    def updateSliderLabel(self, slider, label):
        value = slider.value() / 100.0
        label.setText(f'{value:.2f}')

    def updateSliderLabelZ(self, slider, label):
        value = 2 ** (slider.value() / 100.0)
        label.setText(f'{value:.2f}')

    def reset(self):
        self.slider_cam_x.setValue(0)
        self.slider_cam_y.setValue(0)
        self.slider_cam_z.setValue(0)
        self.slider_cam_off.setValue(85)
        self.camCfgCheckbox.setChecked(True)