import shutil
import sys
import time

import cv2
import os
import numpy as np
import imageio
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()


        self.videoLabel = QLabel()
        self.videoLabel.setText('Video display area')
        self.videoLabel.setStyleSheet("background-color: grey")
        self.videoLabel.setMinimumWidth(512)
        self.videoLabel.setMinimumHeight(320)
        self.videoLabel.setAlignment(Qt.AlignCenter)


        self.generateButton = QPushButton('>>> Generate video <<<')
        # self.generateButton.clicked.connect(self.playVideo)

        self.pauseButton = QPushButton('⏸ Pause')
        self.pauseButton.clicked.connect(self.pauseVideo)
        self.pauseButton.setEnabled(False)


        self.saveButton = QPushButton('Save video')
        self.saveButton.clicked.connect(self.saveVideo)
        self.saveButton.setEnabled(False)

        self.clearButton = QPushButton('Clear')
        self.clearButton.clicked.connect(self.clearVideo)
        
        self.video_np = []  # 新增一个成员变量来存储视频帧列表
        self.current_frame_idx = 0  # 新增一个变量来跟踪当前的帧索引

        # self.isGenerationCompleted = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.cap = None
        self.isPlaying = True


    @pyqtSlot()
    def playVideo(self, video_np):
        self.video_np = video_np  # 设置视频帧列表
        self.current_frame_idx = 0  # 重置帧索引
        self.pauseButton.setEnabled(True)
        self.saveButton.setEnabled(True)

        self.timer.start(125)  # 1000/8fps = 125ms

    def updateFrame(self):
        if self.current_frame_idx < len(self.video_np):
            frame = self.video_np[self.current_frame_idx]
            self.displayImage(frame)
            self.current_frame_idx += 1
        else:
            self.current_frame_idx = 0  # 如果到达列表末尾，则循环播放


    def pauseVideo(self):
        if self.isPlaying:
            self.timer.stop()
            self.pauseButton.setText('▶ Resume')
        else:
            self.timer.start(125)  # 1000/8fps = 125ms
            self.pauseButton.setText('⏸ Pause')
        self.isPlaying = not self.isPlaying

    def saveVideo(self):
        if self.video_np is None:
            return
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Video", "video",
                                                  "MP4 Files (*.mp4);;All Files (*)", options=options)
        if fileName:
            if not fileName.endswith('.mp4'):
                fileName += '.mp4'
            imageio.mimsave(fileName, self.video_np, format='mp4', fps=8)


    def displayImage(self, img):
        img = np.ascontiguousarray(img) # Ensure that the frame is contiguous.
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(outImage))

    def clearVideo(self):
        self.timer.stop()
        self.video_np = []
        self.videoLabel.clear()
        self.videoLabel.setText('Video display area')
        self.pauseButton.setText('⏸ Pause')
        self.isPlaying = True

        self.pauseButton.setEnabled(False)
        self.saveButton.setEnabled(False)



    def drawProgressBar(self, progress):
        # 确保进度值在0到1之间
        progress = max(0, min(1, progress))
        pixmap = QPixmap(self.videoLabel.size())
        pixmap.fill(Qt.transparent)  # 填充为透明`
        # 创建QPainter实例用于绘制
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 255, 255, 127))  # 设置画刷为半透明的红色
        painter.setPen(Qt.NoPen)  # 不需要边框
        # 计算进度条的宽度和高度
        progressBarWidth = pixmap.width() * progress
        progressBarHeight = pixmap.height()
        # 绘制进度条
        self.videoLabel.clear()
        painter.drawRect(0, 0, progressBarWidth, progressBarHeight)
        painter.end()  # 完成绘制
        # 将绘制好的pixmap设置回label
        self.videoLabel.setPixmap(pixmap)

    def pp(self):
        for i in range(100):
            time.sleep(0.1)
            self.drawProgressBar(i/100)
            QApplication.processEvents()  # 允许Qt处理其他事件



    '''
        @pyqtSlot()
    def playVideo2(self):

        # if self.cap is None: # and self.isGenerationCompleted:  # Only if the video is not already loaded
        self.pauseButton.setEnabled(True)
        self.saveButton.setEnabled(True)
        self.cap = cv2.VideoCapture(".out/__output__.mp4")  # Replace with your video file path
        self.timer.start(125)  # 1000/8fps = 125ms

    def updateFrame2(self):
        ret, frame = self.cap.read()
        if ret:
            self.displayImage2(frame)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video

    def saveVideo2(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Video", "video",
                                                  "MP4 Files (*.mp4);;All Files (*)", options=options)
        if fileName:
            if not fileName.endswith('.mp4'):
                fileName += '.mp4'
            shutil.copy(self.video_path, fileName)
    def displayImage2(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # rows[0],cols[1],channels[2]
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        self.videoLabel.setPixmap(QPixmap.fromImage(outImage))
        self.videoLabel.setAlignment(Qt.AlignCenter)

    def clearVideo2(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.videoLabel.clear()
        self.videoLabel.setText('Video display area')
        self.pauseButton.setText('⏸ Pause')
        self.isPlaying = True

        self.pauseButton.setEnabled(False)
        self.saveButton.setEnabled(False)
    '''