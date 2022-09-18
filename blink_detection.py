from git_eyeimgdata import EyeData

class BlinkDetector(EyeData):
    def __init__(self, img_org, img_gray, landmark):
        super().__init__(img_org, img_gray,landmark)



    # def updatedMinMaxValue(self, min, max, new_val):
    #     min = new_val if new_val < min else min
    #     max = new_val if new_val > max else max
    #
    #     return min, max

