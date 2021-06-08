import cv2                # pip install opencv-python
import mediapipe as mp    # pip install mediapipe
import numpy as np

####################################################
import sys
from PyQt5 import QtCore, QtWidgets, QtGui    # pip install pyqt5
from PyQt5.QtGui import QMovie

class Sticker(QtWidgets.QMainWindow):
    def __init__(self, img_path, xy, size=1.0, on_top=False, text=""):
        super(Sticker, self).__init__()
        self.timer = QtCore.QTimer(self)
        self.img_path = img_path
        self.xy = xy
        self.from_xy = xy
        self.from_xy_diff = [0, 0]
        self.to_xy = xy
        self.to_xy_diff = [0, 0]
        self.speed = 60
        self.direction = [0, 0] # x: 0(left), 1(right), y: 0(up), 1(down)
        self.size = size
        self.on_top = on_top
        self.localPos = None
        self.text = text
        self.textLabel = None
        self.textAnim = None
        self.textColor = None
        self.movieRPS = []
        self.label = None

        self.setupUi()
        self.show()

    # 마우스 놓았을 때
    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.to_xy_diff == [0, 0] and self.from_xy_diff == [0, 0]:
            pass
        else:
            self.walk_diff(self.from_xy_diff, self.to_xy_diff, self.speed, restart=True)

    # 마우스 눌렀을 때
    def mousePressEvent(self, a0: QtGui.QMouseEvent):
        self.localPos = a0.localPos()

    # 드래그 할 때
    def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
        self.timer.stop()
        self.xy = [(a0.globalX() - self.localPos.x()), (a0.globalY() - self.localPos.y())]
        self.move(*map(int, self.xy))

    def walk(self, from_xy, to_xy, speed=60):
        self.from_xy = from_xy
        self.to_xy = to_xy
        self.speed = speed

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.__walkHandler)
        self.timer.start(1000 / self.speed)

    # 초기 위치로부터의 상대적 거리를 이용한 walk
    def walk_diff(self, from_xy_diff, to_xy_diff, speed=60, restart=False):
        self.from_xy_diff = from_xy_diff
        self.to_xy_diff = to_xy_diff
        self.from_xy = [self.xy[0] + self.from_xy_diff[0], self.xy[1] + self.from_xy_diff[1]]
        self.to_xy = [self.xy[0] + self.to_xy_diff[0], self.xy[1] + self.to_xy_diff[1]]
        self.speed = speed
        if restart:
            self.timer.start()
        else:
            self.timer.timeout.connect(self.__walkHandler)
            self.timer.start(1000 / self.speed)

    def __walkHandler(self):
        if self.xy[0] >= self.to_xy[0]:
            self.direction[0] = 0
        elif self.xy[0] < self.from_xy[0]:
            self.direction[0] = 1

        if self.direction[0] == 0:
            self.xy[0] -= 1
        else:
            self.xy[0] += 1

        if self.xy[1] >= self.to_xy[1]:
            self.direction[1] = 0
        elif self.xy[1] < self.from_xy[1]:
            self.direction[1] = 1

        if self.direction[1] == 0:
            self.xy[1] -= 1
        else:
            self.xy[1] += 1

        self.move(*self.xy)

    def setupUi(self):
        vbox = QtWidgets.QVBoxLayout()
        hbox = QtWidgets.QHBoxLayout()

        text = QtWidgets.QLabel(self.text)
        #text = MyLabel(self.text)
        #text._set_color(QtGui.QColor("#fff"))
        text.setAlignment(QtCore.Qt.AlignCenter)
        text.setStyleSheet(
            "font-size: 20px;"
            "font-weight: bold;"
            "background-color: #fff;"
            "color: #000;"
            "padding: 5px 10px;"
            "border-radius: 5px;"
            "border: 1px solid #000;"
        )
        self.textLabel = text

        # https://zetcode.com/pyqt/qpropertyanimation/
        #self.anim = QtCore.QPropertyAnimation(text, b"color")
        #self.anim.setDuration(2000)
        #self.anim.setLoopCount(-1)
        #rainbow = [ # http://jsfiddle.net/tovic/RjyHm/
        #    "white", "red", "orange", "gold", "yellow", "yellowgreen",
        #    "green", "cyan", "skyblue", "violet", "magenta", "indigo"
        #]
        #for i, v in enumerate(rainbow):
        #    self.anim.setKeyValueAt(i/len(rainbow), QtGui.QColor(v))
        #self.anim.setEndValue(QtGui.QColor("white"))
        #self.anim.start()

        hbox.addStretch(1)
        hbox.addWidget(text)
        hbox.addStretch(1)

        centralWidget = QtWidgets.QWidget(self)
        centralWidget.setLayout(vbox)

        self.setCentralWidget(centralWidget)

        flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint if self.on_top else QtCore.Qt.FramelessWindowHint)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        label = QtWidgets.QLabel(centralWidget)
        label.setAlignment(QtCore.Qt.AlignCenter)

        for img_path in self.img_path:
            movie = QMovie(img_path)
            movie.start()
            movie.stop()
            w = int(movie.frameRect().size().width() * self.size)
            h = int(movie.frameRect().size().height() * self.size)
            movie.setScaledSize(QtCore.QSize(w, h))
            movie.start()
            self.movieRPS.append(movie)

        vbox.addWidget(label)
        vbox.addLayout(hbox)

        label.setMovie(self.movieRPS[0])
        self.label = label

        #self.setWindowIcon(QtGui.QIcon('icon.ico'))
        self.setGeometry(self.xy[0], self.xy[1], w, h)
        #self.resize(w, h)

    # 더블 클릭 했을 때
    def mouseDoubleClickEvent(self, e):
        print("더블클릭!")
        pass

    def modifyText(self, text=""):
        self.textLabel.setText(str(text))

    def win_to(self, idx=""):
        if idx == 0:   # rock
            self.label.setMovie(self.movieRPS[1])
        elif idx == 5: # paper
            self.label.setMovie(self.movieRPS[2])
        elif idx == 9: # scissors
            self.label.setMovie(self.movieRPS[0])
        else:
            self.label.setMovie(self.movieRPS[0])

# 어플리케이션 시작
app = QtWidgets.QApplication(sys.argv)
a = Sticker(['img/r.png', 'img/p.png', 'img/s.png'], xy=[0, 0], size=0.5, on_top=True, text="인공지능")
####################################################

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # 어플리케이션에 idx를 넘긴다.
            a.win_to(idx)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Human', img)
    if cv2.waitKey(1) == ord('q'):
        break
