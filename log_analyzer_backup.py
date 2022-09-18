import numpy as np
from git_eyeimgdata import EyeData

class LogProcessor(EyeData):
    def __init__(self, data, master_log, minMax):
        super(LogProcessor, self).__init__()
        # super(LogProcessor, self).__init__(img_org, img_gray, landmarks)
        # self.
    # def __init__(self, img_org, img_gray, landmarks, master_log, minMax):
    #     super().__init__(img_org, img_gray, landmarks)
        self.master_log = master_log
        self.minMax = minMax
    #     self.log_e2f, self.log_w2h, self.log_puplx, self.log_puply = self.master_log
        self.data = self.data_output
        self.new_data = self.dataConverted_as2DigitInteger()

        # UPDATED LOG
        self.new_master_log = self.updatedMasterLog()
        self.modes = self.get_allModes_byRatio_fromLog(self.new_master_log)


    """COMMON STARTS HERE"""


    def multiply_then_round_int(self, number, multiply):
        number_new = round(number*multiply, 0)
        number_new = np.nan if np.isnan(number_new) else int(number_new)
        return number_new

    def dataConverted_as2DigitInteger(self):
        rto_e2f, rto_w2h, rto_puplx, rto_puply= self.data

        rto_e2f = [self.multiply_then_round_int(item, 1) for item in rto_e2f]
        rto_w2h = [self.multiply_then_round_int(item, 10) for item in rto_w2h]
        rto_puplx = [self.multiply_then_round_int(item, 100) for item in rto_puplx]
        rto_puply = [self.multiply_then_round_int(item, 100) for item in rto_puply]

        return rto_e2f, rto_w2h, rto_puplx, rto_puply

    def update_count_inDictLog(self, log, value):
        log[value] = log[value]+1 if value in log else 1

        return log

    def mode_fromDictionary(self, log):
        # print(f"log = {log}")
        mode = max(log.items(), key=lambda x: x[1])[0] if len(log)!=0 else np.nan

        return mode


    # def get_std_inDictLog(self, log):


    def update_1x2Data_in2dLog(self, log, data):
        for i, lg in enumerate(log):
            lg = self.update_count_inDictLog(lg, data[i])

        return log

    def get_allModes_byRatio_fromLog(self, master_log):
        modes_all = []
        for lg_rto in master_log: #rto_2ef, rto_puplx, rto_puply
            modes_rto = []
            for lg_side in lg_rto:
                mode = self.mode_fromDictionary(lg_side)
                modes_rto.append(mode)
            modes_all.append(modes_rto)

        return modes_all


    """COMMON ENDS HERE"""

    def updatedMasterLog(self):
        data = self.data
        # data = self.new_data
        # self.data_input = [self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply]
        # self.data_input = [[L, R], [L, R], [L, R], [L, R]]
        master_log = self.master_log
        #self.master_log = self.log_e2f, self.log_w2h, self.log_puplx, self.log_puply
        #log_m = [[{L}, {R}], [{L}, {R}], [{L}, {R}], [{L}, {R}]]

        for idx_data, log_rto in enumerate(master_log):
            data_new = data[idx_data]
            log_rto = self.update_1x2Data_in2dLog(log_rto, data_new)

        return master_log
    
    # def modes_fromUpdatedMasterLog(self):
    #     for log_rto in self.updatedMasterLog():
            