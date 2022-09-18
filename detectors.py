import cv2
import dlib
import numpy as np

class Camera:
    # Color
    __RED = (0, 0, 255)
    __GREEN = (0, 255, 0)
    __BLUE = (255, 0, 0)
    __YELLOW = (0, 255, 255)
    __SKYBLUE = (255, 255, 0)
    __PURPLE = (255, 0, 255)
    __WHITE = (255, 255, 255)
    __BLACK = (0, 0, 0)

    # Resolution
    # __PIXELS = [(1920, 1080), (640, 480), (256, 144), (320, 240), (480, 360)]
    __PIXELS = [(1280, 720), (640, 480), (256, 144), (320, 240), (480, 360)]
    def __init__(self, path=0):
        self.PIXEL_NUMBER = 0
        self.RES_W = self.__PIXELS[0][0]
        self.RES_H = self.__PIXELS[0][1]
        print(f"pixel set ({self.__PIXELS[0][1]},{self.__PIXELS[0][0]})")
        try:
            self.cap = cv2.VideoCapture(path)
        except:
            print("Error opening video stream or file")

    def getFrameResize2ndarray(self, frame):
        return np.array(imutils.resize(frame, width=self.RES_W, height=self.RES_H))

    def getFrame2array(self, frame):
        pass

    def setPixels(self, resolution):
        print(f"H:{self.__PIXELS[resolution][1]} W{self.__PIXELS[resolution][0]}")
        return self.__PIXELS[resolution]

    def getRed(self):
        return self.__RED

    def getGreen(self):
        return self.__GREEN

    def getBlue(self):
        return self.__BLUE

    def getYellow(self):
        return self.__YELLOW

    def getSkyblue(self):
        return self.__SKYBLUE

    def getPurple(self):
        return self.__PURPLE

    def getWhite(self):
        return self.__WHITE

    def getBlack(self):
        return self.__BLACK

class FaceDetector:
    def __init__(self):
        # HOG(Histogram of Oriented Gradients) 특성
        # dlib에 있는 얼굴 검출기 사용
        self.face_detector = dlib.get_frontal_face_detector()

    def front_detection(self, squares):
        print("front_detection")
        most_front_detection_index = 0
        max_size_area = 0
        for i, sq in enumerate(squares):
            curr_area = (sq.right() - sq.left()) * (sq.bottom() - sq.top())
            if curr_area > max_size_area:
                max_size_area = curr_area
                most_front_detection_index = i
        return most_front_detection_index

    def get_facebox_2dot_form(self, image, upscale=0):
        face_detect = self.face_detector(image, upscale)
        if face_detect:
            if len(face_detect) > 1:
                return face_detect[self.front_detection(face_detect)]
            else:
                return face_detect[0]
        return None

class MarkDetector:
    __NOTHING = list(range(0, 0))

    __ALL = list(range(0, 68))

    __FACE_OUTLINE = list(range(0, 17))

    __LEFT_EYEBROW = list(range(17, 22))
    __RIGHT_EYEBROW = list(range(22, 27))

    __NOSE = list(range(27, 36))

    __LEFT_EYE = list(range(36, 42))
    __RIGHT_EYE = list(range(42, 48))

    __MOUTH_OUTLINE = list(range(48, 60))
    __MOUTH_INLINE = list(range(60, 68))

    __MARK_INDEX = __RIGHT_EYE + __LEFT_EYE + __MOUTH_INLINE
    # __MARK_INDEX = __ALL

    def __init__(self, save_model="./assets/shape_predictor_68_face_landmarks.dat"):

        print(f"stub loading facial landmark predictor {save_model}...")
        self.shape_predictor = dlib.shape_predictor(save_model)
        print(f"complete loading facial landmark predictor!")

    def get_marks(self, image, detect):
        # print(type(detect)) # <class '_dlib_pybind11.rectangles'> 인데?
        # detect인자는 <class _dlib_pybind11.full_object_detection> 타입으로 입력해야함 이라고 오류 나는데
        # 다른 파일에서는 <class '_dlib_pybind11.rectangle'> 로 shape_predictor가능한데? 뭐지? 날 화나게 하는건가?
        # 와 코드랑 싸울뻔 했다 (결론 본인이 멍청 했던 걸로)
        shape = self.shape_predictor(image, detect)
        return shape

    def draw_marks(self, image, marks, color=(225, 255, 255)):
        if isinstance(marks, np.ndarray):
            for i in self.__MARK_INDEX:
                cv2.circle(image, (marks[i][0], marks[i][1]), 1, color, -1, cv2.LINE_AA)
        elif isinstance(marks, dlib.full_object_detection):
            for i in self.__MARK_INDEX:
                cv2.circle(image, (marks.part(i).x, marks.part(i).y), 1, color, -1, cv2.LINE_AA)

    def draw_box(self, image, rect, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in rect:
            cv2.rectangle(image,
                          (box[0], box[1]), (box[2], box[3]),
                          box_color, 3)

    def full_object_detection_to_ndarray(self, full_object):
        result = [[p.x, p.y] for p in full_object.parts()]
        result = np.array(result)
        return result

    def landMarkPutOnlyRectangle(self, img, rect):
        """
        :param img: 원본 이미지
        :param rect: 얼굴 detection : dlib._dlib_pybind11.rectangle
        :return: rect에 roi된 이미지, rect roi에 맞춘 랜드마크
        """
        landmark = self.get_marks(img, rect)
        rectImg = img[rect.top():rect.bottom(), rect.left():rect.right()]
        # print(x1, y1)

        landmark = self.full_object_detection_to_ndarray(landmark)  # (x, y)
        landmark[:, 0] -= rect.left()
        landmark[:, 1] -= rect.top()

        return rectImg, landmark

    def pyrUpWithLandmark(self, img, landmark, iterator=1):
        """
        :param img: 원본 이미지
        :param landmark: 원본 이미지에 매칭되는 랜드마크
        :param iterator: 피라미드 횟수(default=1)
        :return: 피라미드 업 한 이미지, 피라미드 업 된 이미지에 맞춘 랜드마크
        """
        sizeUpImg = img.copy()
        for _ in range(iterator):
            sizeUpImg = cv2.pyrUp(sizeUpImg)

        sizeUpLandmark = landmark * 2**iterator
        return sizeUpImg, sizeUpLandmark

    def changeMarkIndex(self, key):
        # TODO: 랜드마크 보여주는거 변경하는거 뭐지? ㅎ
        if key == 1:
            self.__MARK_INDEX = self.__NOTHING
        elif key == 2:
            self.__MARK_INDEX = self.__LEFT_EYEBROW + self.__RIGHT_EYEBROW
        elif key == 3:
            self.__MARK_INDEX = self.__LEFT_EYE + self.__RIGHT_EYE
        elif key == 4:
            self.__MARK_INDEX = self.__NOSE
        elif key == 5:
            self.__MARK_INDEX = self.__MOUTH_INLINE + self.__MOUTH_OUTLINE
        elif key == 6:
            self.__MARK_INDEX = self.__FACE_OUTLINE


class Tracker:
    def __init__(self):
        self.frame_counter = 0
        self.track_number = 0
        self.tracker = None

    def find_tracker(self, img, fd, re=False):
        box = fd.get_facebox_2dot_form(img)     # FaceDetector
        tk = dlib.correlation_tracker()    # correlation_traker타입 객채 생성
        if box is not None:
            if re:
                self.frame_counter = 0
                tk.start_track(img, box)
                self.tracker = tk
            else:
                # rect = dlib.rectangle(box.left(), box.top(), box.right(), box.bottom())
                tk.start_track(img, box)  # 얼굴 감지한 네모를 트래킹 하기 시작
                self.tracker = tk  # 트래킹 할 정보 추가
                self.track_number += 1
            return self.dlib_corr_tracker_to_rectangle(self.tracker)

    def tracking(self, img):
        self.frame_counter += 1
        self.tracker.update(img)    #트래킹 갱신
        # 여기서 tracker는 find_tracker에서 correlation_tracker 타입으로 append되었기 때문에
        # rectangle 타입으로 바꿔서 넘겨 주어야 한다
        rect = self.dlib_corr_tracker_to_rectangle(self.tracker)
        # self.draw_rectangle(img, rect)
        return rect

    def dlib_corr_tracker_to_rectangle(self, corr):
        pos = corr.get_position()
        rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        return rect

    def draw_rectangle(self, frame, tracker, color=(0, 255, 0)):
        cv2.rectangle(frame, (tracker.left(), tracker.top()),
                      (tracker.right(), tracker.bottom()), color, 3)

    def getRectangle(self, img, facedetector):
        detect = None
        if self.track_number == 0:
            self.find_tracker(img, facedetector)
            detect = None
        else:
            if self.frame_counter == 60:
                detect = self.find_tracker(img, facedetector, re=True)
            else:
                detect = self.tracking(img)
        return detect