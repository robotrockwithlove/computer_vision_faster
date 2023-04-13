import cv2
import sys
import time
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QMainWindow, QSizePolicy, QPushButton

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        cap = cv2.VideoCapture(self.filename)
        #cap.set(cv2.CAP_PROP_FPS, 30)  # set frame rate to 30 fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        delay = int(1000 / (fps * 1.3))

        while True:
            ret, cv_img = cap.read()
            if ret:
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img)
                cv2.waitKey(delay)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(True)

        # create the video capture thread
        self.thread = VideoThread('short_street.avi')

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)

        # create the start/stop button
        self.button = QPushButton('Start', self.image_label)
        self.button.clicked.connect(self.toggle_video)

        # start the thread
        self.thread.start()

        self.setCentralWidget(self.image_label)

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    @pyqtSlot()
    def toggle_video(self):
        if not self.thread.isRunning():
            self.thread.start()
            self.button.setText('Stop')
        else:
            self.thread.terminate()
            self.thread.wait()
            self.button.setText('Start')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())