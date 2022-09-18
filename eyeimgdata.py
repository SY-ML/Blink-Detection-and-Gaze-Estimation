import numpy as np
import cv2
import matplotlib.pyplot as plt

class EyeData():
    def __init__(self, img_org, img_gray, landmarks):
        """
        @param img_org: original image in which results will be shown 
        @param img_gray: gray scale image
        @param landmarks: 68 landmarks of face
        """
        # IMAGES
        self.img_show, self.img_input = img_org, img_gray
        ## LANDMARK
        # landmarks of eyes (left, right)
        self.ldmk_eyeL, self.ldmk_eyeR = self.ldmk_eyes = [landmarks[36:42], landmarks[42:48]]
        self.ldmk_face = landmarks[0:17] 
        
        ## MISCELLANEOUS
        self.num_pxPd = 2 # number of padding pixel to top, bottom, left, and right of the cropped eye images
        
        ## PROCESSED EYE IMAGES
        self.msk_eyeL, self.msk_eyeR = self.msk_eyes = self.images_ofEyes_withWhiteBackground()
        self.vals_tholds = self.optimalValues_ofPupils()
        self.imgs_thold = self.imagesThreshold_withOptimalValue()

        ## CONTOURS
        self.coord_pupil = self.contourData_ofEyes()[0]
        self.data_contours = self.contourData_ofEyes()[1]

        ## DIMENSION & AREA
        self.dim_eyes = self.widthAndHeight_ofEyes()
        self.rto_w2h = self.ratios_eyeHeightToWidth()
        self.rto_e2f = self.ratios_eyeAreaToFaceArea()

        ## PUPIL CENTEROID
        self.rto_puplx = self.ratios_pupilX_toImageWidth()
        self.rto_puply = self.ratios_pupilY_toImageHeight()

        self.data_output = [self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply]
    """ COMMON STARTS HERE"""

    def minX_maxX_minY_maxY_ofLandmarks(self, landmarks):
        """
        @param landmarks: landmarks of a part in face
        @return: [min X, max X, min Y, max Y] 
        """
        # sort x and y separately (ascending)
        sorted_byX, sorted_byY = sorted(landmarks, key=lambda ldmk: ldmk[0]), sorted(landmarks, key=lambda ldmk: ldmk[1])
        range_xy = np.array([sorted_byX[0][0], sorted_byX[-1][0], sorted_byY[0][1], sorted_byY[-1][1]])
        return range_xy

    def fromTo_range_ofLandmarks(self, landmarks):
        px_pd = self.num_pxPd
        x_min, x_max, y_min, y_max = self.minX_maxX_minY_maxY_ofLandmarks(landmarks)
        list_fromTo = [y_min - px_pd, y_max + px_pd, x_min - px_pd, x_max + px_pd]
        arr_fromTo = np.array(list_fromTo)

        return arr_fromTo

    def coord_topLeft_ofLandmarks(self, landmarks):
        """
        @param landmarks: landmark of a part in face 
        @return: (x, y) of the top left location
        """
        list_topLeft = [self.fromTo_range_ofLandmarks(landmarks)[2::-2]]
        # arr_fromTo returns [y_min - px_pd, y_max + px_pd, x_min - px_pd, x_max + px_pd]/
        # [y_from, y_to, x_from, x_to]
        # What are needed : x_from (left), y_min (top)
        # array[2::-2] loads x_from, y_from
        arr_leftTop = np.array(list_topLeft)
        return arr_leftTop

    def imageCrop_byLandmark(self, image, landmark):
        """
        @param image: Image to crop from 
        @param landmark: 
        @return: cropped image of the landmark
        """
        y_from, y_to, x_from, x_to = self.fromTo_range_ofLandmarks(landmark) #range from/to which the image is cropped.
        img_crop = image[y_from:y_to, x_from:x_to] # image cropped
        return img_crop

    

    def mask_withBlackBackground_byLandmark(self, image, landmark):
        """
        @param image: Image for the mask to be like
        @param landmark: 
        @return: mask image with white background
        """
        mask_blckBg = np.zeros_like(image, dtype=np.uint8) # 
        mask = cv2.fillConvexPoly(mask_blckBg, landmark,
                                  255)  # connect the points of the landmark and fill it with white.

        img_bit = cv2.bitwise_and(image, image, mask=mask)
        bg_white = np.ones_like(image, dtype='uint8') * 255
        cv2.bitwise_not(bg_white, bg_white, mask=mask)
        img_bit = bg_white + img_bit

        return img_bit


    def area_ofLandmark(self, landmark):
        """
        Returns the area of landmark
        @param landmark:
        @return: Int. Number of pixels
        """
        img_crp = self.imageCrop_byLandmark(self.img_input, landmark)
        img_mask = self.mask_withBlackBackground_byLandmark(img_crp, landmark)
        area = np.count_nonzero(img_mask)

        return area
   
    """ COMMON ENDS HERE"""



    def images_ofEyes_withWhiteBackground(self):
        """
        Returns images of eyes with white background
        @return: [ image of left eye, image of right eye ]
        """
        img_output = []
        
        for ldmk_eye in self.ldmk_eyes: # Load landmarks
            # Crop eye images and create masks for them
            img_crp = self.imageCrop_byLandmark(self.img_input, ldmk_eye)
            xy_topLft = self.coord_topLeft_ofLandmarks(ldmk_eye)
            ldmkCvt_eye = np.subtract(ldmk_eye, xy_topLft) # Convert landmarks to the ones in the cropped image
            mask_blckBg = np.zeros_like(img_crp, dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask_blckBg, ldmkCvt_eye, 255)

            def histogram(image):
                hist = cv2.calcHist(image, [0], None, [256], [0, 256])
                return hist

            # f, ax = plt.subplots(4, 2, figsize=(18, 8))

            # hist0 = histogram(img_crp)
            # ax[0][0].imshow(img_crp, cmap='gray')
            # ax[0][1].plot(hist0)
            # ax[0][1].set_title("ORG")

            #Preprocess: GaussianBlur > HistEqualization > MedianBlur
            # img_crp = cv2.GaussianBlur(img_crp, (3, 3), 0)
            # # img_crp = cv2.medianBlur(img_crp, 3)
            # hist1 = histogram(img_crp)
            # ax[1][0].imshow(img_crp, cmap='gray')
            # ax[1][1].plot(hist1)
            # ax[1][1].set_title("STEP1")
            #
            # img_crp = cv2.equalizeHist(img_crp)
            # hist2 = histogram(img_crp)
            # ax[2][0].imshow(img_crp, cmap='gray')
            # ax[2][1].plot(hist2)
            # ax[2][1].set_title("STEP2")

            # img_crp = cv2.GaussianBlur(img_crp, (3, 3), 0)
            img_crp = cv2.GaussianBlur(img_crp, (3, 3), 0)
            img_crp = cv2.medianBlur(img_crp, 3)
            img_crp = cv2.equalizeHist(img_crp)
            # img_crp = cv2.medianBlur(img_crp, 3)
            # img_crp = cv2.bilateralFilter(img_crp, -1, 10, 50, 50 )
            # img_crp = cv2.blur(img_crp, (3,3))
            # img_crp = cv2.medianBlur(img_crp, 3)
            # hist3 = histogram(img_crp)
            # ax[3][0].imshow(img_crp, cmap='gray')
            # ax[3][1].plot(hist3)
            # ax[3][1].set_title("STEP3")
            #
            # plt.show()
            # Mask and leave the background white
            img_bit = cv2.bitwise_and(img_crp, img_crp, mask=mask)
            bg_white = np.ones_like(img_bit, dtype='uint8') * 255
            cv2.bitwise_not(bg_white, bg_white, mask=mask)
            img_bit = bg_white + img_bit
            img_output.append(img_bit)

        return img_output

    def optimalValue_ofImage(self, image):
        """
        
        @param image: input image to get an optimal threshold from
        @return: value of brightness with the first least count following the mode of the brightness value
        """
        # Convert brightness values over the median to 255 
        median = np.median(image)
        img_underMedian = np.where(image<median, image, 255)
        
        unique, count = np.unique(img_underMedian, return_counts=True)
        unique, count = unique[:-1], count[:-1] # value 255 and its counts removed
        idx, num_counts = np.argmax(count), len(count) # index of the mode, num of elements in count

        while(idx+1 < num_counts): 
            if count[idx+1] <=count[idx]: # Continue if count of the next value is less
                idx+=1
            else: break

        optim_val = unique[idx] # optimal value = Value greater than the mode and with the least count.

        return optim_val
    """ COMMON ENDS HERE"""

    def widthAndHeight_ofEyes(self):
        output = []
        for ldmk_eye in self.ldmk_eyes:
            minX, maxX, minY, maxY = self.minX_maxX_minY_maxY_ofLandmarks(ldmk_eye)
            width, height = maxX - minX, maxY - minY
            dim = (width, height)
            output.append(dim)

        return output

    def optimalValues_ofPupils(self):
        """
        Returns optimal values of the pupils' brightness
        @return: [optimal value(L-pupil), optimal value(R-pupil]
        """

        arr_optimThold = np.array([], dtype='uint8')
        for img_mask in self.msk_eyes:
            optim_thold = self.optimalValue_ofImage(img_mask)
            arr_optimThold = np.append(arr_optimThold, optim_thold)

        return arr_optimThold
    #
    # def imagesThreshold_withOptimalValue(self):
    #     list_imgsThold = []
    #
    #     for i, img_mask in enumerate(self.msk_eyes):
    #         # self.imshow_Larger(f"MASK{i}", img_mask) # MASK IMG
    #         thold = self.vals_tholds[i]
    #         _, img = cv2.threshold(img_mask, thold, 255, cv2.THRESH_OTSU)
    #         # _, img = cv2.threshold(img_mask, thold, 255, cv2.THRESH_OTSU, cv2.THRESH_TOZERO_INV)
    #         # _, img = cv2.threshold(img_mask, thold, 255, cv2.THRESH_TOZERO_INV)
    #
    #         img = cv2.erode(img, (3,3))
    #         img = cv2.dilate(img, (3,3))
    #         # self.imshow_Larger(f"PRCD{i}", img) # THOLD IMG
    #         list_imgsThold.append(img)
    #
    #     return list_imgsThold

    def imagesThreshold_withOptimalValue(self):
        list_imgsThold = []

        for i, img_mask in enumerate(self.msk_eyes):
            # self.imshow_Larger(f"MASK{i}", img_mask) # MASK IMG
            thold = self.vals_tholds[i]
            _, img = cv2.threshold(img_mask, thold, 255, cv2.THRESH_BINARY_INV)

            img = cv2.erode(img, (3,3), iterations=2)
            img = cv2.dilate(img, (3,3), iterations=2)
            # self.imshow_Larger(f"PRCD{i}", img) # THOLD IMG
            list_imgsThold.append(img)

        return list_imgsThold
    #
    # def contour_ofEyes(self):
    #     output = []
    #     for img_thold in self.imgs_thold:
    #         contours, _ = cv2.findContours(img_thold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         contours = sorted(contours, key=lambda x: cv2.contourArea(x),
    #                           reverse=True)  # Print the contour with the biggest area first.
    #         for cnt in contours:
    #             (x, y, w, h) = cnt_data = cv2.boundingRect(cnt)
    #             xy_ctr = x_ctr, y_ctr = ((2 * x + w) / 2, (2 * y + h) / 2)
    #             cv2.drawContours(img_thold, [cnt], -1, (255, 255, 255), 1)
    #             output.append(cnt_data)
    #             cv2.imshow("cnt", img_thold)
    #             break
    #     return output

    def contourData_ofEyes(self):
        output_centorid = []
        output_allSpots = []

        for img_thold in self.imgs_thold:
            contours, _ = cv2.findContours(img_thold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                contour_all = max(contours, key=lambda x: cv2.contourArea(x))
                (x, y, w, h) = cv2.boundingRect(contour_all)
                ctr_x = round((2*x+w)/2, 0)
                ctr_y = round((2*y+h)/2, 0)
                output_centorid.append((int(ctr_x), int(ctr_y)))
                output_allSpots.append(contour_all)
                # cv2.drawContours(img_thold, [contour], -1, (255, 0, 0), 1)
                # output.append(contour_data)
                # cv2.imshow("Cnt", img_thold)
            except:
                output_centorid.append((np.nan, np.nan))
                output_allSpots.append(None)
        # print(f"output_centorid = {output_centorid}")
        # print(f"output_allSpots = {output_allSpots}")

        return output_centorid, output_allSpots


    def ratios_pupilX_toImageWidth(self):
        images = self.imgs_thold # images after threshholding
        coord_pupil = self.coord_pupil # coord_pupil [(x, y), (x, y) ]
        output = []
        for side in range(2):
            ctr_x = coord_pupil[side][0] # x of the centroid
            wth = images[side].shape[1] # width of the cropped eye image
            ratio = ctr_x/wth
            output.append(ratio)

        return output

    def ratios_pupilY_toImageHeight(self):
        images = self.imgs_thold # images after threshholding
        coord_pupil = self.coord_pupil # coord_pupil [(x, y), (x, y) ]
        output = []
        for side in range(2):
            ctr_y = coord_pupil[side][1] # x of the centroid
            hgt = images[side].shape[0] # width of the cropped eye image
            ratio = ctr_y/hgt
            output.append(ratio)

        return output


    def ratios_eyeHeightToWidth(self):
        output = []
        for ldmk_eye in self.ldmk_eyes:
            minX, maxX, minY, maxY = self.minX_maxX_minY_maxY_ofLandmarks(ldmk_eye)
            width, height = maxX-minX, maxY-minY
            ratio = width/height if height != 0 else np.nan
            # ratio = height/width if height != 0 else np.nan
            output.append(ratio)
        return output
    
    def ratios_eyeAreaToFaceArea(self):

        area_face = self.area_ofLandmark(self.ldmk_face)
        output = []
        for ldmk_eye in self.ldmk_eyes:
            area_eye = self.area_ofLandmark(ldmk_eye)
            # ratio = round(area_eye/area_face,0)
            ratio = area_face/area_eye if area_face !=0 else np.nan
            # ratio = area_eye/area_face if area_face !=0 else np.nan
            output.append(ratio)

        return output


"""
"""
#
# class Blog(EyeData):
#     def __init__(self, img_org, img_gray, landmarks):
#         super().__init__(img_org, img_gray, landmarks)
#
#     """
#     1. 영상: 크롭이미지/마스킹이미지/HE이미지/THOLD 이미지/컨투어적용이미지
#     2. 영상찍기
#     """
#
#
#     def write_img(self, img):
#         path = "C:/Dropbox/DMS/1.DMS_FINAL/blog_img/"
#         list_files = os.listdir(path)
#         num_files = len(list_files)+1
#         fileName = f"img{num_files}.jpg"
#
#         path_final = path+str(fileName)
#         cv2.imwrite(path_final, img)
#         print(f"{fileName} saved in the following path: {path}")
#
#     def img_org(self, show=False, save=False):
#         img_org = self.img_show
#
#         return img_org
#
#     def img_crp(self):
#         imgs_crp = []
#         for ldmk_eye in self.ldmk_eyes:
#             img_crp = self.imageCrop_byLandmark(self.img_input, ldmk_eye)
#             imgs_crp.append(img_crp)
#
#         return imgs_crp
#
#     def img_msk(self):
#         imgs_msk = self.img_msk()
#
#         return imgs_msk
#
#     def img_msk_step1(self): # CROP
#         img_output = []
#         for ldmk_eye in self.ldmk_eyes:
#             xy_topLft = self.coord_topLeft_ofLandmarks(ldmk_eye)
#             ldmk_eye = np.subtract(ldmk_eye, xy_topLft)
#             img_crp = self.imageCrop_byLandmark(self.img_input, ldmk_eye)
#             mask_blckBg = np.zeros_like(img_crp, dtype=np.uint8)
#             mask = cv2.fillConvexPoly(mask_blckBg, ldmk_eye, 255)
#             img_output.append(mask)
#
#         return img_output
#
#     def img_msk_step2(self, show=False):
#         img_output = []
#         for ldmk_eye in self.ldmk_eyes:
#             img_crp = self.imageCrop_byLandmark(self.img_input, ldmk_eye)
#             xy_topLft = self.coord_topLeft_ofLandmarks(ldmk_eye)
#             ldmk_eye = np.subtract(ldmk_eye, xy_topLft)
#             mask_blckBg = np.zeros_like(img_crp, dtype=np.uint8)
#             mask = cv2.fillConvexPoly(mask_blckBg, ldmk_eye, 255)
#             #STEP 2
#             img_crp = cv2.GaussianBlur(img_crp, (3,3), 0)
#             img_crp = cv2.equalizeHist(img_crp)
#             img_crp = cv2.medianBlur(img_crp, 3)
#
#             img_bit = cv2.bitwise_and(img_crp, img_crp, mask=mask)
#             bg_white = np.ones_like(img_bit, dtype='uint8') * 255
#             cv2.bitwise_not(bg_white, bg_white, mask=mask)
#             img_bit = bg_white + img_bit
#             # img_bit = cv2.medianBlur(img_bit, 3)
#             if show is True:
#                 self.qucik_calcHistogram(img_bit, 254)
#
#             # MEDIAN BLUR
#
#             img_output.append(img_bit)
#
#         return img_output
#
#     def img_thold(self):
#         img_output = []
#         msks = self.img_msk_step2()
#         for img_msk in msks:
#             thold = self.optimalValue_ofImage(img_msk)
#             _, img = cv2.threshold(img_msk, thold, 255, cv2.THRESH_BINARY_INV)
#             img = cv2.erode(img, (3,3), iterations=2)
#             img = cv2.dilate(img, (3,3), iterations=2)
#             img_output.append(img)
#
#         return img_output
#
#     def img_contour(self):
#         img_output = []
#         img_crp = self.img_crp()
#         img_msk = self.img_msk_step2()
#
#         for i, img in enumerate(self.img_thold()):
#             contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             # contours = max(contours, key=lambda x: cv2.contourArea(x))
#             contours = sorted(contours, key=lambda x: cv2.contourArea(x),
#                               reverse=True)  # Print the contour with the biggest area first.
#             for cnt in contours:
#                 (x, y, w, h) = cv2.boundingRect(cnt)
#                 xy_ctr = x_ctr, y_ctr = ((2 * x + w) / 2, (2 * y + h) / 2)
#                 cv2.drawContours(img_crp[i], [cnt], -1, (255, 255, 255), 1)
#                 break
#             # cv2.circle(img_crp[i], (round(x_ctr, 0), round(y_ctr)), round(min(w, h) / 4), (255, 255, 255))
#             img_output.append(img_crp[i])
#         return img_output
#
#     def write_img_asVideo(self, img, fileName):
#         fourcc = cv2.VideoWriter_fourcc(*"DIVX")
#         fps = 30
#         size = (1920, 1080)
#         out = cv2.VideoWriter(f"{fileName}.avi", fourcc, fps, size)
#         out.write(img)
