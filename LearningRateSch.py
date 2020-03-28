
import math
class CosineDecaySchedule :

    def __init__(self,lrMin,lrMax,tMaxIni,factor,lrDecayRate):
        """
        :param lrMin: The min learning rate in one schedule
        :param lrMax: The max learning rate in one schedule
        :param tMaxIni: The initial max training times in one schedule
        :param factor: increase tMaxIni by multiply this factor at every restart
        :param lrDecayRate : The decay rate of lrMax
        """
        self.lrMin = lrMin
        self.lrMax = lrMax
        self.curTT = 0.
        self.trainingTimes = tMaxIni
        self.factor = factor
        self.lrDecayRate = lrDecayRate

    def calculateLearningRate(self):
        if self.curTT > self.trainingTimes:
            self.__restart()
        lrC = float(self.lrMin) + 0.5 * (self.lrMax - self.lrMin) \
              * (1. + math.cos(self.curTT / self.trainingTimes * math.pi))
        return lrC

    def step(self):
        self.curTT += 1

    def __restart(self):
        self.lrMax = self.lrMax * self.lrDecayRate
        self.curTT = 0
        self.trainingTimes = self.trainingTimes * self.factor

