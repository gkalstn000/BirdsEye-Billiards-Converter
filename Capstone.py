import sys
import cv2 as cv
import numpy as np
import urllib.request
import numpy as np
import cordinate.Billiards_Detect_test as bd
import cordinate.point_order as po
import object_detection.cut_obj as co
import trans.imgwarp2 as iw
import trans.show_result as sr
import time

from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *



#Main_#
class MainDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUI()
        
    #LayOut2
    def setupUI(self):
        self.Running=True
        #self.Stop = False
        
        self.setGeometry(200, 200, 500, 500)
        self.setWindowTitle("Frame")
        self.setWindowIcon(QIcon('icon.png'))
        #self.pushButton1 = QPushButton(self)
        #self.pushButton.setGeometry(QRect(20, 20, 100, 40))
        
        
        #label.setGeometry(QRect(10, 20, 10, 10))
        self.label = QLabel()
        self.label2 = QLabel()
        self.pushButton1= QPushButton("Start")
        self.pushButton2= QPushButton("Stop")
        self.pushButton3= QPushButton("Cancel")
        self.pushButton1.clicked.connect(self.Play_Video_clicked)
        self.pushButton2.clicked.connect(self.Stop_Video_button)
        self.pushButton3.clicked.connect(self.Exit_button)
        #layout = QGridLayout()
        #layout.addWidget(self.label, 0, 0)
        #layout.addWidget(self.label2, 0, 1)
        #layout.addWidget(self.pushButton1, 1, 0)
        #layout.addWidget(self.pushButton2, 1, 1)
        #layout.addWidget(self.pushButton3, 1, 2)
        
        
        
        Hlayout = QHBoxLayout()
        Hlayout.addStretch(1)
        Hlayout.addWidget(self.label)
        Hlayout.addStretch(1)
        Hlayout.addWidget(self.label2)
        Hlayout.addStretch(1)
        Hlayout2 = QHBoxLayout()
        Hlayout2.addWidget(self.pushButton1)
        Hlayout2.addWidget(self.pushButton2)
        Hlayout2.addWidget(self.pushButton3)
        Vlayout = QVBoxLayout()
        Vlayout.addLayout(Hlayout)
        Vlayout.addStretch(1)
        Vlayout.addLayout(Hlayout2)
        
        self.setLayout(Vlayout)
    #VideoPlay_button
    def Play_Video_clicked_Origin(self) :
         
        if self.Running == True :
            capture = cv.VideoCapture(self.file)
            ret, image = capture.read()
            h, w = image.shape[:2]
            self.h, self.w = h, w
        else :
            self.Running = True
            capture = self.capture
            h, w = self.h, self.w
        self.loop = QEventLoop()
        
        while self.Running :
            start = time.time()
            ret, frame = capture.read()
            self.capture = capture
            if ret :
                #후에 편집영상
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                canny = cv.Canny(frame_gray, 50, 100)
                
                
                qImg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                qcanny = QImage(canny.data, w, h, canny.strides[0], QImage.Format_Grayscale8)
      
                
                pixmap = QPixmap.fromImage(qImg)
                pixcmap = QPixmap.fromImage(qcanny)#현재 Canny영상,  후에 편집 영상 들어갈 자리.
                
                
                self.label.setPixmap(QPixmap(pixmap))
                self.label2.setPixmap(QPixmap(pixcmap))
                
                QTimer.singleShot(30, self.loop.quit)
                self.loop.exec_()
            else :
                capture = cv.VideoCapture(self.file)
            print("cost time :", time.time() - start)
        
    def Play_Video_clicked(self) :

        
        if self.Running == True :
            capture = cv.VideoCapture(self.file)
            ret, image = capture.read()
            h, w = image.shape[:2]
            self.h, self.w = h, w
        else :
            self.Running = True
            capture = self.capture
            h, w = self.h, self.w
        self.loop = QEventLoop()
        
        while self.Running :
            
            ret, frame = capture.read()
            self.capture = capture
            if ret :
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                result_dict = co.img_cut(self.file)

                image_np = result_dict['image_np']
                points = result_dict['points']
                
                
                start = time.time() 
                result = np.array(bd.Detecting(image_np))
                result = po.point_order(result)
                all_points = [(result[0][0], result[0][1]),
             (result[1][0], result[1][1]),
             (result[2][0], result[2][1]),
             (result[3][0], result[3][1])]

                all_points.append((points[0][0], points[0][1]))
                all_points.append((points[1][0], points[1][1]))
                all_points.append((points[2][0], points[2][1]))
                print("cost time :", time.time() - start)
                
                final_image = iw.warp(all_points)
                final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
                height,width = final_image.shape[:2]
                #height, width = int(height/4), int(width/4)
                
                #final_image=cv.resize(final_image, (width, height))

                
                qImg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                qchimg = QImage(final_image.data, width, height, final_image.strides[0], QImage.Format_RGB888)
      
                
                pixmap = QPixmap.fromImage(qImg)
                pixcmap = QPixmap.fromImage(qchimg)#현재 Canny영상,  후에 편집 영상 들어갈 자리.
                
                
                self.label.setPixmap(QPixmap(pixmap))
                self.label2.setPixmap(QPixmap(pixcmap))
                
                QTimer.singleShot(30, self.loop.quit)
                self.loop.exec_()
            else :
                capture = cv.VideoCapture(self.file)
            
    def Stop_Video_button(self) :
        if self.Running == True :
            print("Stop")
            self.Running = False
            
    def Exit_button(self) :
        print("Exit")
        self.Running =False
        self.close()
        
#Home_Menu
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
    #Layout
    def setupUI(self):
        self.setGeometry(800, 200, 500, 300)
        self.setWindowTitle("Capstone v1.0")
        self.setWindowIcon(QIcon('icon.png'))

        self.pushButton = QPushButton("Notebook")
        self.pushButton2 = QPushButton("Video")
        self.pushButton3 = QPushButton("Cancel")
        
        self.pushButton.clicked.connect(self.pushButtonClicked_Main_Notebook)
        self.pushButton2.clicked.connect(self.pushButtonClicked_Main_Video)
        self.pushButton3.clicked.connect(self.pushButtonClicked_Menu_Exit)
        
        #V : 수직, H : 수평 
        layout = QVBoxLayout()
        layout.addWidget(self.pushButton)
        layout.addWidget(self.pushButton2)
        layout.addStretch(1)
        layout.addWidget(self.pushButton3)
        
        self.setLayout(layout)

    def pushButtonClicked_Main_Notebook(self):
        dlg = MainDialog()
        dlg.file = 0
        dlg.exec_()
        
        
    def pushButtonClicked_Main_Video(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', "",
                                            "Video Files(*.mp4);; Python Files(*.py);; All Files(*.*) ", '/home')
        if fname[0]:
            dlg = MainDialog()
            dlg.file = fname[0]
            dlg.exec_()
            print(fname[1])
        else:
            QMessageBox.about(self, "Warning", "파일을 선택하지 않았습니다.")
            
        
        
    def pushButtonClicked_Menu_Exit(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
