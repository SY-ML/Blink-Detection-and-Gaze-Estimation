import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from git_detectors import Tracker, FaceDetector, MarkDetector, Camera
from git_eyeimgdata import EyeData
from git_log_analyzer import LogProcessor
from git_data_obtainer import DataObtainer
from git_blink_detection import BlinkDetector
# path_video = "/dataset_ver1.1.mp4"
path_video = "WIN_20220526_15_33_19_Pro.mp4"
# path_video = "IMG_8721.MOV"
### Classes for VideoCapture, FaceDetection, and 68-LandmarkDetection

cm = Camera(path=path_video)  # path를 안하면 카메라 하면 영상
tk = Tracker()
fd = FaceDetector()
md = MarkDetector()

def show_images(name, images, times = 20):
    for idx, img in enumerate(images):
        h, w = img.shape
        img_rszd = cv2.resize(img, dsize=(w*times, h*times))
        cv2.imshow(f"{name}-{idx}", img_rszd)

def updatedMinMax(minMax, new_vals):
    for idx, side in enumerate(minMax):
        new_val = new_vals[idx]
        side[0] = new_val if new_val < side[0] else side[0]
        side[1] = new_val if new_val > side[1] else side[1]

    return minMax


minMax_e2f = [[float('inf'), float('-inf')], [float('inf'), float('-inf')]]
minMax_w2h = [[float('inf'), float('-inf')], [float('inf'), float('-inf')]]
minMax_puplx = [[float('inf'), float('-inf')], [float('inf'), float('-inf')]]
minMax_puply = [[float('inf'), float('-inf')], [float('inf'), float('-inf')]]
minMax_blink = [minMax_e2f, minMax_w2h]
minMax_pupil = [minMax_puplx, minMax_puply]

sp_blink = 100
sp_pupil = 300


log_e2f = [{}, {}]
log_w2h = [{}, {}]
log_puplx = [{}, {}]
log_puply = [{}, {}]

log_blink = [log_e2f, log_w2h]
log_gaze = [log_puplx, log_puply]

# master_log = [log_e2f, log_w2h, log_puplx, log_puply]


cnt_frm=0
rto_e2f = np.array([])
rto_w2h = np.array([])
rto_puplx = np.array([])
rto_puply = np.array([])




while cm.cap.isOpened():
    ret, img_org = cm.cap.read()  # 영상 프레임 받기
    key = cv2.waitKey(1)
    if key == 27:
        break
    if ret:
        img_rszd = cv2.resize(img_org, dsize=(1280, 720))
        img_gray = cv2.cvtColor(img_rszd, cv2.COLOR_BGR2GRAY) # Convert img_org to grayscale
        rect = tk.getRectangle(img_gray, fd)

        if rect is not None:
            landmarks = md.get_marks(img_gray, rect)
            landmarks = md.full_object_detection_to_ndarray(landmarks)
            md.draw_marks(img_org, landmarks, color=cm.getRed())

            # For the first 300 frames (10 seconds) - Data Obtaining
            if cnt_frm <= 30:
                ed = EyeData(img_rszd, img_gray, landmarks)
                minMax_e2f = updatedMinMax(minMax_e2f, ed.rto_e2f)
                minMax_w2h = updatedMinMax(minMax_w2h, ed.rto_w2h)

                cv2.imshow("ORG", img_rszd)
                cnt_frm +=1

            elif cnt_frm <= 200:

                ed = EyeData(img_rszd, img_gray, landmarks)
                data_eye = ed.data_output
                lp = LogProcessor(data_eye, minMax_blink, log_blink,minMax_pupil, log_gaze)
                minMax_puplx = updatedMinMax(minMax_puplx, ed.rto_puplx)
                minMax_puply = updatedMinMax(minMax_puply, ed.rto_puply)

                print(f"blink(normalized) = {lp.data_blink_normalized} ")
                print(f"blink(log) = {lp.log_blink_updated}")
                if cnt_frm >100:
                    lp.smoothData_inLog(lp.log_blink_updated[0][0])
                cv2.imshow("ORG", img_rszd)
                cnt_frm +=1


            else: # A
                # rto_minMax = [minMax_e2f, minMax_w2h, minMax_puplx, minMax_puply]
                ed = EyeData(img_rszd, img_gray, landmarks)

                # lp = LogProcessor(data_eye, master_log, rto_minMax)


                cv2.imshow("ORG", img_rszd)
                cnt_frm +=1


                # master_log = lp.updatedMasterLog()

            """
            #크롭된 이미지
            crp = [ed.imageCrop_byLandmark(ed.img_input, ldmk_eye) for ldmk_eye in ed.ldmk_eyes]
            """

            """
            #RATIO 시각화
            # rto1 = ed.ratios_eyeHeightToWidth()[0]*10
            # rto2 = ed.ratios_eyeAreaToFaceArea()[0]
            rto_px = ed.ratios_pupilX_toImageWidth()[0]*100
            rto_py = ed.ratios_pupilY_toImageHeight()[0]*100
            # print(f"rto_px = {np.isnan(rto_px)}")
            rto_px = np.nan if np.isnan(rto_px) else int(rto_px)
            rto_py = np.nan if np.isnan(rto_py) else int(rto_py)



            rto_w2h = np.append(rto_w2h, int(ed.ratios_eyeHeightToWidth()[0]*10))
            rto_e2f = np.append(rto_e2f, int(ed.ratios_eyeAreaToFaceArea()[0]))
            rto_puplx = np.append(rto_puplx, rto_px)
            rto_puply = np.append(rto_puply, rto_py)
            # rto_puplx = np.append(rto_puplx, int(ed.ratios_pupilX_toImageWidth()[0]*100))
            # rto_puply = np.append(rto_puply, int(ed.ratios_pupilY_toImageHeight()[0]*100))

            if cnt_frm>0 and cnt_frm%500 == 0:
                f, ax = plt.subplots(1, 4, figsize=(18, 8))
                ax[0].plot(rto_w2h, label="rto_w2h")
                ax[0].plot(rto_e2f, label="rto_w2h")
                ax[0].set_title("Ratios by frame")
                ax[0].set_xlabel("frame")
                ax[0].set_ylabel("ratio")
                ax[0].legend()

                ax[2].plot(rto_puplx, label="rto_puplx")
                ax[2].plot(rto_puply, label="rto_puplY")
                ax[2].set_title("Ratios by frame")
                ax[2].set_xlabel("frame")
                ax[2].set_ylabel("ratio")
                ax[2].legend()



                sns.kdeplot(rto_w2h, ax=ax[1], label="rto_w2h")
                sns.kdeplot(rto_e2f, ax=ax[1], label="rto_e2f")
                ax[1].set_title("Ratio Distribution")
                ax[1].set_xlabel("ratio")
                ax[1].set_ylabel("density")
                ax[1].legend()

                sns.kdeplot(rto_puplx, ax=ax[3], label="rto_puplx")
                sns.kdeplot(rto_puply, ax=ax[3], label="rto_puply")
                ax[3].set_title("Ratio Distribution")
                ax[3].set_xlabel("ratio")
                ax[3].set_ylabel("density")
                ax[3].legend()

                f.suptitle(f"frame: {cnt_frm}")
                plt.tight_layout()
                plt.show()
            # """


    else:
        cv2.destroyAllWindows()
        cm.cap.release()
        break


