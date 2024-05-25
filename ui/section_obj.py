# from PyQt5.QtMultimedia import QMediaPlayer
# from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QCheckBox, QVBoxLayout, QGroupBox, QHBoxLayout,
                             QPushButton, QGraphicsView, QSlider, QFileDialog,
                             QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem,
                             QGraphicsPathItem,
                             QLineEdit, QLabel)
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt5.QtGui import QMouseEvent, QPen, QBrush, QColor, QPolygonF, QPainterPath
import sys, math
import numpy as np


class Arrow(QGraphicsLineItem):
    def __init__(self, start, end):
        super().__init__(QLineF(start, end))
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.start = start
        self.end = end
        self.drawArrowHead()

    def drawArrowHead(self):
        line = QLineF(self.end, self.start)
        arrowSize = 10.0

        angle = line.angle()

        # 计算箭头头部的两个点
        p1 = self.end + QPointF(arrowSize * math.cos(math.radians(angle + 160)),
                                arrowSize * math.sin(math.radians(angle + 160)))
        p2 = self.end + QPointF(arrowSize * math.cos(math.radians(angle - 160)),
                                arrowSize * math.sin(math.radians(angle - 160)))

        # 添加箭头头部
        self.arrowHead = QGraphicsPolygonItem(QPolygonF([self.end, p1, p2]), self)
        self.arrowHead.setBrush(QBrush(QColor(255, 0, 0)))



class ObjSection(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setFixedSize(512+5, 320+5)
        self.setSceneRect(0, 0, 512, 320)  # 设置场景大小

        self.startPoint = None
        self.tempRect = None
        self.rects = []  # save current box (pair)
        self.inter_points = []  # save user-defined points for connecting boxes
        self.isDrawingEnabled = True

        self.currentArrow = None  # 用于跟踪当前的箭头图形项
        self.current_path = None # store the current path connecting start-end boxes
        self.currentGraphicsItems = []  # 用于跟踪当前box对的所有相关图形项

        self.obj_assets = []  # 存储所有box对及其路径和箭头的列表
        self.boxColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 不同box对的颜色

        self.box_matrix = []  # store the final obj bbox coordinates (x1y1x2y2)

        ### Widigts
        self.enableSwitch = QCheckBox("Enable Object Motion Control")
        self.enableSwitch.setChecked(False)  # 默认设置为non勾选状态
        self.enableSwitch.toggled.connect(self.toggleAvailability)  # 连接信号槽


        self.addObjBtn = QPushButton("Add Object")
        self.addObjBtn.clicked.connect(self.registerCurrentObj)

        self.finishBtn = QPushButton("Finish")
        self.finishBtn.clicked.connect(self.finishDrawing)
        self.finishBtn.hide() # save all boxes to self.box_matrix.

        self.clearAllBtn = QPushButton("Clear")
        self.clearAllBtn.clicked.connect(self.clearAll)

        ## save Box Sequence button
        self.saveButton = QPushButton("Save box sequence as .npy file", self)
        self.saveButton.clicked.connect(self.saveBoxSequence)


        # Attention amplification weight
        self.slider_creg = QSlider(Qt.Horizontal)
        self.slider_creg.setValue(25) # default
        self.slider_creg.setMinimum(0)
        self.slider_creg.setMaximum(50)
        # self.slider_creg.setMinimumWidth(5)
        self.value_creg = QLabel(f'{25}')
        self.slider_creg.valueChanged.connect(lambda: self.updateLabel(self.slider_creg, self.value_creg))
        self.layout_creg = QHBoxLayout()
        self.layout_creg.addWidget(QLabel("Attention amplification weight:"))
        self.layout_creg.addWidget(self.slider_creg)
        self.layout_creg.addWidget(self.value_creg)
        
        # Attention amplification timestep
        self.slider_tau = QSlider(Qt.Horizontal)
        self.slider_tau.setValue(95) # default
        self.slider_tau.setMinimum(80)
        self.slider_tau.setMaximum(100)
        # self.slider_tau.setMinimumWidth(5)
        self.value_tau = QLabel(f'{0.95:.2f}')
        self.slider_tau.valueChanged.connect(lambda: self.updateLabel_div100(self.slider_tau, self.value_tau))
        self.layout_tau = QHBoxLayout()
        self.layout_tau.addWidget(QLabel("Attention amplification timestep:"))
        self.layout_tau.addWidget(self.slider_tau)
        self.layout_tau.addWidget(self.value_tau)

        self.toggleAvailability(self.enableSwitch.isChecked())


    def toggleAvailability(self, checked):
        self.clearAll()
        self.isDrawingEnabled = checked
        self.addObjBtn.setEnabled(checked)
        self.addObjBtn.setEnabled(checked)
        self.finishBtn.setEnabled(checked)
        self.clearAllBtn.setEnabled(checked)
        self.saveButton.setEnabled(checked)
        self.slider_tau.setEnabled(checked)
        self.slider_creg.setEnabled(checked)

    def updateLabel(self, slider, label):
        value = slider.value()
        label.setText(f'{value}')
    
    def updateLabel_div100(self, slider, label):
        value = slider.value() / 100.0
        label.setText(f'{value:.2f}')
        
        
    def mousePressEvent(self, event: QMouseEvent):
        if self.isDrawingEnabled and event.button() == Qt.LeftButton and len(self.rects) < 2: # 2 boxes at most
            self.startPoint = self.mapToScene(event.pos())
            colorIndex = len(self.obj_assets) % len(self.boxColors)  # 根据当前box对数选择颜色
            color = self.boxColors[colorIndex]
            pen = QPen(QColor(*color), 2)
            brush = QBrush(QColor(*color, 50))  # 半透明填充
            self.tempRect = QGraphicsRectItem(QRectF(self.startPoint, self.startPoint))
            self.tempRect.setPen(pen)
            self.tempRect.setBrush(brush)
            self.scene.addItem(self.tempRect)

        # elif len(self.rects) > 2 and event.button() == Qt.LeftButton:
        #     print("you cannot draw a third box")
        elif len(self.rects) == 2 and event.button() == Qt.RightButton:
            # 添加关键点并绘制曲线
            keyPoint = self.mapToScene(event.pos())
            self.inter_points.append(keyPoint)
            self.drawSmoothCurve()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.isDrawingEnabled and self.startPoint is not None:
            endPoint = self.mapToScene(event.pos())
            rect = QRectF(self.startPoint, endPoint).normalized()
            self.tempRect.setRect(rect)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.isDrawingEnabled and event.button() == Qt.LeftButton:
            self.rects.append(self.tempRect)
            self.tempRect = None
            self.startPoint = None
            colorIndex = (len(self.obj_assets) % len(self.boxColors))  # 选择颜色索引
            if len(self.rects) == 2:  # 每次完成一对box后
                self.drawInitialArrow(colorIndex)  # 使用颜色索引
                # self.registerCurrentObj()
            # if len(self.rects) > 2:
            #     print("you cannot draw a third box")

    def drawInitialArrow(self, colorIndex):
        if len(self.rects) < 2:
            return
        color = self.boxColors[colorIndex]
        pen = QPen(QColor(*color), 2)
        startRect = self.rects[-2].rect()
        endRect = self.rects[-1].rect()
        startCenter = startRect.center()
        endCenter = endRect.center()
        path = QPainterPath(startCenter)
        path.lineTo(endCenter)

        pathItem = self.scene.addPath(path, pen)
        self.drawArrow(path, pen)

        self.current_path = path
        self.currentGraphicsItems.append(pathItem)

    def drawArrow(self, path, pen):
        if path.elementCount() < 2: # or not self.currentArrow:
            return  # 路径中的点不足以绘制箭头

        # 清除旧箭头
        if self.currentArrow:
            for arrowPart in self.currentArrow:
                self.scene.removeItem(arrowPart)
            self.currentArrow = []

        # 获取路径的最后两个点，以确定箭头的方向
        lastPoint = path.elementAt(path.elementCount() - 1)
        preLastPoint = path.elementAt(path.elementCount() - 2)
        line = QLineF(QPointF(preLastPoint.x, preLastPoint.y), QPointF(lastPoint.x, lastPoint.y))

        # # 使用colorIndex来确定箭头颜色
        # color = self.boxColors[colorIndex]
        # pen = QPen(QColor(*color), 2)

        # 计算箭头头部的两个点
        arrowSize = 10.0
        angle = math.atan2(-line.dy(), line.dx())

        p1 = line.p2() - QPointF(math.sin(angle + math.pi / 3) * arrowSize,
                                 math.cos(angle + math.pi / 3) * arrowSize)
        p2 = line.p2() - QPointF(math.sin(angle + math.pi - math.pi / 3) * arrowSize,
                                 math.cos(angle + math.pi - math.pi / 3) * arrowSize)

        # 绘制箭头
        arrowLine1 = self.scene.addLine(QLineF(line.p2(), p1), pen)
        arrowLine2 = self.scene.addLine(QLineF(line.p2(), p2), pen)

        # 更新当前箭头
        self.currentArrow = [arrowLine1, arrowLine2]

    def drawSmoothCurve(self):
        if len(self.rects) < 2:
            return  # 确保有起始和结束box

        # 清除清除之前的曲线
        for item in self.currentGraphicsItems:
            self.scene.removeItem(item)
        self.currentGraphicsItems.clear()

        colorIndex = (len(self.obj_assets) % len(self.boxColors))
        color = self.boxColors[colorIndex]
        pen = QPen(QColor(*color), 2)

        startRect = self.rects[-2].rect()
        endRect = self.rects[-1].rect()
        startCenter = startRect.center()
        endCenter = endRect.center()

        points = [startCenter] + self.inter_points + [endCenter]  # 总关键点列表

        path = self.generateCatmullRomSpline(points, 0.5)
        if path:
            pathItem = self.scene.addPath(path, pen)
            self.drawArrow(path, pen)             # 在曲线尾端绘制箭头
            self.currentGraphicsItems.append(pathItem)  # 记录当前box对的曲线图形项
            self.current_path = path

    def generateCatmullRomSpline(self, points, tightness):
        """
        生成Catmull-Rom样条曲线路径。
        :param points: 控制点列表。
        :param tightness: 曲线的紧密度。
        :return: QPainterPath对象。
        """
        if len(points) < 2:
            return None

        path = QPainterPath(points[0])
        for i in range(len(points) - 1):
            p0 = points[i - 1] if i - 1 >= 0 else points[0]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2] if i + 2 < len(points) else points[-1]

            for t in range(1, 21):  # 生成20个点来近似每个Catmull-Rom段
                t = t / 20.0
                tt = t * t
                ttt = tt * t

                q1 = -tightness * ttt + 2.0 * tightness * tt - tightness * t
                q2 = (2.0 - tightness) * ttt + (tightness - 3.0) * tt + 1
                q3 = (tightness - 2.0) * ttt + (3.0 - 2.0 * tightness) * tt + tightness * t
                q4 = tightness * ttt - tightness * tt

                x = p0.x() * q1 + p1.x() * q2 + p2.x() * q3 + p3.x() * q4
                y = p0.y() * q1 + p1.y() * q2 + p2.y() * q3 + p3.y() * q4

                path.lineTo(QPointF(x, y))

        return path

    def interpolateBoxSequenceAlongPath(self, path, rects, num_boxes):
        if path is None and len(rects)==1: # single box
            startRect = rects[0].rect()
            topLeftX = max(startRect.x()/self.width(), 0)
            topLeftY = max(startRect.y()/self.height(), 0)
            bottomRightX = min((startRect.x()+startRect.width())/self.width(), 1)
            bottomRightY = min((startRect.y()+startRect.height())/self.height(), 1)
            box_matrix = [[topLeftX, topLeftY, bottomRightX, bottomRightY]] * num_boxes
            return np.array(box_matrix)

        ## compute box centers along the path
        centers = []
        for i in range(num_boxes):
            percent = i/(num_boxes-1)
            point = path.pointAtPercent(percent)
            centers.append((point.x(), point.y()))

        ## interpolate box sequence
        startRect = rects[0].rect()
        endRect = rects[1].rect()
        # 获取首尾box的长宽
        startWidth = startRect.width()
        startHeight = startRect.height()
        endWidth = endRect.width()
        endHeight = endRect.height()
        # 初始化numpy矩阵
        box_matrix = np.zeros((num_boxes, 4))
        # 计算每个box的位置和大小
        for i, center in enumerate(centers):
            width = np.linspace(startWidth, endWidth, num_boxes)[i]
            height = np.linspace(startHeight, endHeight, num_boxes)[i]
            topLeftX = center[0] - width / 2
            topLeftY = center[1] - height / 2
            bottomRightX = center[0] + width / 2
            bottomRightY = center[1] + height / 2
            # 归一化坐标
            topLeftX = max(topLeftX/self.width(), 0)
            topLeftY = max(topLeftY/self.height(), 0)
            bottomRightX = min(bottomRightX/self.width(), 1)
            bottomRightY = min(bottomRightY/self.height(), 1)

            box_matrix[i] = [topLeftX, topLeftY, bottomRightX, bottomRightY]

        return box_matrix

    def registerCurrentObj(self):
        if len(self.obj_assets) < 2:  # 最多允许3 objs
            self.isDrawingEnabled = True
            if self.rects:  # save current obj info
                self.obj_assets.append({'path': self.current_path, 'rect':self.rects, 'inter_points':self.inter_points})
            self.currentArrow = None  # 重置箭头跟踪
            self.inter_points = []  # 重置关键点列表
            self.currentGraphicsItems = []
            self.rects = []
            self.current_path = None
        else:
            print("Maximum number of objects reached.")

    def saveBoxSequence(self):
        self.registerCurrentObj()
        num_boxes = 24 # self.slider.value()
        for i, obj_asset in enumerate(self.obj_assets):
            box_matrix = self.interpolateBoxSequenceAlongPath(obj_asset['path'],
                                                           obj_asset['rect'],
                                                           num_boxes)
            # print(box_matrix)

            # 保存矩阵
            fileName, _ = QFileDialog.getSaveFileName(self, f"Save Box Sequence {i+1}", "", "Numpy Files (*.npy)")
            if fileName:
                np.save(fileName, box_matrix)

    def clearAll(self):
        # 清除所有box对及其轨迹
        self.scene.clear()
        self.obj_assets = []
        self.currentArrow = None  # 重置箭头跟踪
        self.inter_points = []  # 重置关键点列表
        self.currentGraphicsItems = []
        self.rects = []
        self.box_matrix=[]
        self.isDrawingEnabled = True

    def enableDrawing(self):
        self.isDrawingEnabled = True

    def finishDrawing(self, num_boxes=24):
        self.registerCurrentObj()
        self.isDrawingEnabled = False
        self.box_matrix = []
        for i, obj_asset in enumerate(self.obj_assets):
            box_matrix = self.interpolateBoxSequenceAlongPath(obj_asset['path'],
                                                           obj_asset['rect'],
                                                           num_boxes)
            print(box_matrix)
            self.box_matrix.append(box_matrix)

        # 这里可以添加其他在完成绘制后需要执行的代码
        print("Finished drawing.")

