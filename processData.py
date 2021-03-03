from caserec.recommenders.item_recommendation.base_item_recommendation import BaseItemRecommendation
from caserec.utils.process_data import ReadFile
from caserec.utils.process_data import WriteFile

inputFile = "data/ml-100k/u.data"

class processData(object):
    def __int__(self, inputFile, outTestFile = 1, outTrainFile=1):
        self.inputFile = inputFile
        self.outTestFile = outTestFile
        self.outTrainFile = outTrainFile
        self.trainSets = ReadFile(inputFile).read()
    def export_data(self):
        print(self.trainSets)