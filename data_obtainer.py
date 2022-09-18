from git_eyeimgdata import EyeData

class DataObtainer(EyeData):
    def __init__(self, img_org, img_gray, landmarks):
        super().__init__(img_org, img_gray, landmarks)

    """COMMON STARTS HERE"""
    def tuple_floatMinMax(self):
        return (float('inf'), float('-inf'))

    def updatedMinMax(self, min, max, new_val):
        min = new_val if new_val < min else min
        max = new_val if new_val > max else max

        return min, max
    """COMMON STARTS HERE"""
