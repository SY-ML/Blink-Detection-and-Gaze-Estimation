import matplotlib.pyplot as plt
import numpy as np
# from git_eyeimgdata import EyeData

class LogProcessor():
    def __init__(self, data,minMax_blink, log_blink, minMax_pupil, log_pupil):
        self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply = data
        self.data_blink = [self.rto_e2f, self.rto_w2h]
        self.data_gaze = [self.rto_puplx, self.rto_puply]

        self.log_blink = log_blink
        self.log_pupil = log_pupil

        self.minMax_e2f, self.minMax_w2h = self.minMax_blink = minMax_blink
        self.minMax_puplx, self.minMax_puply = self.minMax_pupil = minMax_pupil
        self.minMax = [self.minMax_blink, self.minMax_blink]

        self.data_blink_normalized = self.normalization_multipliedBy100(self.minMax_blink, self.data_blink)
        self.log_blink_updated = self.logUpdate(self.data_blink, self.log_blink)


        # self.data_nmzd = self.normalizedRatio_multipliedBy100()

        # UPDATED LOG
        # self.new_master_log = self.updatedMasterLog()
        # self.modes = self.get_allModes_byRatio_fromLog(self.new_master_log)

# class LogProcessor():
#     def __init__(self, data, master_log, minMax):
#         self.data = data
#         self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply = self.data
#         # self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply = self.data
#         self.master_log = master_log
#         self.minMax = minMax
#         self.minMax_e2f, self.minMax_w2h, self.minMax_puplx, self.minMax_puply = self.minMax
#
#         self.data_nmzd = self.normalizedRatio_multipliedBy100()
#
#         # UPDATED LOG
#         self.new_master_log = self.updatedMasterLog()
#         self.modes = self.get_allModes_byRatio_fromLog(self.new_master_log)
#

    """COMMON STARTS HERE"""

    def normalization_multipliedBy100(self, minMax, ratios):
        data = ratios
        # rto_minMax = [minMax_e2f, minMax_w2h, minMax_puplx, minMax_puply]
        #            = [ [[min(L), max(L)], [min(R), max(R)]].... ]

        for idx_rto, rto in enumerate(data):
            for idx_side, val in enumerate(rto): # rto = [value(L), value(R)]
                min, max = minMax[idx_rto][idx_side]
                ratio_nmzd = (val-min)/(max-min)
                rto[idx_side] = np.nan if np.isnan(ratio_nmzd) else int(round(ratio_nmzd*100, 0))
        return data

    #
    # def normalizedRatio_multipliedBy100(self):
    #     data = self.data
    #     # data == [[ratio1L, ratio1R], [ratio2L, ratio2R]... [ratio4L, ratio4R]]
    #     #      == [self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply]
    #     minMax = self.minMax
    #     # rto_minMax = [minMax_e2f, minMax_w2h, minMax_puplx, minMax_puply]
    #     #            = [ [[min(L), max(L)], [min(R), max(R)]].... ]
    #
    #     for idx_rto, rto in enumerate(data):
    #         for idx_side, val in enumerate(rto): # rto = [value(L), value(R)]
    #             min, max = minMax[idx_rto][idx_side]
    #             ratio_nmzd = (val-min)/(max-min)
    #             rto[idx_side] = np.nan if np.isnan(ratio_nmzd) else int(round(ratio_nmzd*100, 0))
    #     return data
    #

    def update_count_inDictLog(self, log, value):
        log[value] = log[value]+1 if value in log else 1

        return log

    def mode_fromDictionary(self, log):
        # print(f"log = {log}")
        mode = max(log.items(), key=lambda x: x[1])[0] if len(log)!=0 else np.nan

        return mode


    def update_1x2Data_in2dLog(self, log, data, sort=True):
        for i, lg in enumerate(log):
            lg = self.update_count_inDictLog(lg, data[i])
            print(f"lg(B) = {lg}")
            if sort is True:
                lg = sorted(lg.items(), key=lambda x: x[0])
                lg = dict(lg)
                print(f"lg(A) = {lg}")

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

    def logUpdate(self, data, log):
        #self.master_log = self.log_e2f, self.log_w2h, self.log_puplx, self.log_puply
        #log_m = [[{L}, {R}], [{L}, {R}], [{L}, {R}], [{L}, {R}]]

        for idx_data, log_rto in enumerate(log):
            data_new = data[idx_data]
            log_rto = self.update_1x2Data_in2dLog(log_rto, data_new)

        return log

    def smoothData_inLog(self, log):
        a = log.items()
        b, c= zip(*a)
        print(f"b = {b}, c = {c}")
        keys, vals = log.keys(), log.values()
        # plt.plot(log.items())
        # plt.show()



    """COMMON ENDS HERE"""
    # def updatedMasterLog(self):
    #     data = self.data_nmzd
    #     # data = self.new_data
    #     # self.data_input = [self.rto_e2f, self.rto_w2h, self.rto_puplx, self.rto_puply]
    #     # self.data_input = [[L, R], [L, R], [L, R], [L, R]]
    #     master_log = self.master_log
    #     #self.master_log = self.log_e2f, self.log_w2h, self.log_puplx, self.log_puply
    #     #log_m = [[{L}, {R}], [{L}, {R}], [{L}, {R}], [{L}, {R}]]
    #
    #     for idx_data, log_rto in enumerate(master_log):
    #         data_new = data[idx_data]
    #         log_rto = self.update_1x2Data_in2dLog(log_rto, data_new)
    #
    #     return master_log

    # def modes_fromUpdatedMasterLog(self):
    #     for log_rto in self.updatedMasterLog():
            