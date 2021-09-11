import time

from caserec.utils.process_data import ReadFile

from src.recommender.processData import convertFile
from src.recommender.process_reducts import process_reduct
from src.recommender.recommenderSystem import recommenderSystem

inputTrainFile = "E:/Khóa luận/SRC/SRC_Recommender2021/recommenderSystem1/data/jester-data-1/50users/data_train.data"
out_resultFile = "E:/Khóa luận/SRC/SRC_Recommender2021/recommenderSystem1/data/export/recommend/recommen.data"
out_reducttFile = "E:/Khóa luận/SRC/SRC_Recommender2021/recommenderSystem1/data/export/recommend/reduct.data"
test_file = "../../data/jester-data-1/50users/data_test.data"
start_time = time.time()


class recommender:
    def __init__(self):
        self.trainSets = ReadFile(inputTrainFile).read()
        self.user_reduct = 3
        # self.user_recommender = 242
        self.cl = {}
        self.dl = {}

    def result(self):
        reducts = {}
        recommen = {}
        for u in self.trainSets['users']:
            if(u <= 1):
                self.user_recommender = u
                reducts.update(process_reduct(self.trainSets, self.user_recommender).process_finding())

                recommen.setdefault(u, recommenderSystem(inputTrainFile=inputTrainFile, test_file=test_file,
                                                         out_resultFile=out_resultFile, k_neighbors=10, \
                                                         reducts=reducts,
                                                         user_recommender=self.user_recommender).recommender())
        print("---------------------------------")
        print(recommen)
        convertFile(out_reducts_file=out_resultFile, data=recommen).WriteResult()
        convertFile(out_reducts_file=out_reducttFile, data=reducts).WriteFile()
        print("--- %s seconds ---" % (time.time() - start_time))


recommender = recommender()
recommender.result()
