from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QCheckBox, QLabel, QVBoxLayout, QGroupBox, QHBoxLayout,
                             QPushButton, QGraphicsView, QSlider, QFileDialog,
                             QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem,
                             QGraphicsPathItem,
                             QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt5.QtGui import QMouseEvent, QPen, QBrush, QColor, QPolygonF, QPainterPath, QTextOption
import sys, math
import numpy as np



class TxtSection(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.enableSwitch = QCheckBox("Enable Interaction")
        self.enableSwitch.setChecked(True)  # 默认设置为勾选状态
        self.enableSwitch.toggled.connect(self.toggleAvailability)  # 连接信号槽

        self.textPromptLabel = QLabel('Text Prompt')
        self.textPrompt = QTextEdit('')
        self.textPrompt.setPlaceholderText('Input your prompt here. \n '
                                           'For object motion control, use * to mark or () to wrap around the object/background word, make sure the background word is the last marked word, \n'
                                           'For example, A tiger* and a bear* walking on (green grassland) \n'
                                           'Accordingly, you should draw trajectories in the object motion canvas on the right-hand side.')
        self.textPrompt.setMinimumWidth(400)
        self.textPrompt.setMinimumHeight(100)
        self.textPrompt.setWordWrapMode(QTextOption.WordWrap)  # 确保启用了自动换行

        self.NegPromptLabel = QLabel('Negative Prompt')
        self.negPrompt = QTextEdit("blurry image, low resolution, disfigured, bad anatomy, deformed body features, poorly drawn face, bad composition")
        self.negPrompt.setMinimumWidth(400)
        self.negPrompt.setMinimumHeight(100)
        self.negPrompt.setWordWrapMode(QTextOption.WordWrap)  # 确保启用了自动换行

        self.slider_cfg = QSlider(Qt.Horizontal)
        self.slider_cfg.setValue(9) # default
        self.slider_cfg.setMinimum(1)
        self.slider_cfg.setMaximum(32)
        self.slider_cfg.setMinimumWidth(400)
        self.value_cfg = QLabel(f'{9}')
        self.slider_cfg.valueChanged.connect(lambda: self.updateLabel(self.slider_cfg, self.value_cfg))


        # 创建重置按钮
        self.textResetButton = QPushButton('Reset')
        self.textResetButton.clicked.connect(self.reset)

    def toggleAvailability(self, checked):
        self.reset()
        self.textPrompt.setEnabled(checked)
        self.negPrompt.setEnabled(checked)
        self.slider_cfg.setEnabled(checked)
        self.textResetButton.setEnabled(checked)


    def reset(self):
        # 重置文本框为默认文本
        self.textPrompt.setPlainText('')
        self.negPrompt.setPlainText('blurry image, low resolution, disfigured, bad anatomy, deformed body features, poorly drawn face, bad composition')
        self.slider_cfg.setValue(9.0)

    def updateLabel(self, slider, label):
        value = slider.value()
        label.setText(f'{value:.1f}')
