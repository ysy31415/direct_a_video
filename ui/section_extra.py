import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView, QWidget, QRadioButton, QVBoxLayout, QHBoxLayout, QLabel, QButtonGroup, \
    QLineEdit, QSlider


class ExtraSection(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.layout_h = QHBoxLayout()
        height_label = QLabel('Height:')
        self.layout_h.addWidget(height_label)

        # 高度单选按钮
        self.height_group = QButtonGroup(self)
        for h in [128, 256, 320]:
            radio = QRadioButton(str(h))
            if h==320:
                radio.setChecked(True)
            self.layout_h.addWidget(radio)
            self.height_group.addButton(radio)

        # 创建宽度选择部分
        self.layout_w = QHBoxLayout()
        width_label = QLabel('Width:')
        self.layout_w.addWidget(width_label)

        # 宽度单选按钮
        self.width_group = QButtonGroup(self)
        for w in [256, 320, 512]:
            radio = QRadioButton(str(w))
            if w==512:
                radio.setChecked(True)
            self.layout_w.addWidget(radio)
            self.width_group.addButton(radio)

        # 创建帧率选择部分
        self.layout_f = QHBoxLayout()
        frame_label = QLabel('Frame:')
        self.layout_f.addWidget(frame_label)

        # 帧率单选按钮
        self.frame_group = QButtonGroup(self)
        for f in [8, 16, 24]:
            radio = QRadioButton(str(f))
            if f==24:
                radio.setChecked(True)
            self.layout_f.addWidget(radio)
            self.frame_group.addButton(radio)

        # Random Seed
        self.layout_seed = QHBoxLayout()
        self.layout_seed.addWidget(QLabel('Seed:'))
        # 单选按钮组
        self.radio_group = QButtonGroup(self)
        self.radio_random = QRadioButton('Random')
        self.radio_fixed = QRadioButton('Fixed')
        self.radio_group.addButton(self.radio_random)
        self.radio_group.addButton(self.radio_fixed)
        self.layout_seed.addWidget(self.radio_random)
        self.layout_seed.addWidget(self.radio_fixed)
        # 数字输入框，用于输入固定种子
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText('42')
        self.seed_input.setText('42')
        self.seed_input.setEnabled(False)  # 默认禁用
        self.layout_seed.addWidget(self.seed_input)
        # 设置默认选择
        self.radio_random.setChecked(True)
        # 连接单选按钮信号
        self.radio_random.toggled.connect(self.toggleSeedInput)
        self.radio_fixed.toggled.connect(self.toggleSeedInput)


        # inference step
        self.slider_step = QSlider(Qt.Horizontal)
        self.slider_step.setValue(50) # default
        self.slider_step.setMinimum(10)
        self.slider_step.setMaximum(100)
        # self.slider_step.setMinimumWidth(5)
        self.value_step = QLabel(f'{50}')
        self.slider_step.valueChanged.connect(lambda: self.updateLabel(self.slider_step, self.value_step))
        self.layout_step = QHBoxLayout()
        self.layout_step.addWidget(QLabel("Inference step:"))
        self.layout_step.addWidget(self.slider_step)
        self.layout_step.addWidget(self.value_step)

    def toggleSeedInput(self):
        # 根据选择启用或禁用输入框
        self.seed_input.setEnabled(self.radio_fixed.isChecked())

    def updateLabel(self, slider, label):
        value = slider.value()
        label.setText(f'{value}')

    def reset(self):
        pass