import itertools
import random
from caserec.utils.process_data import ReadFile

out_reducts_file = '../../data/jester-data-1/data_reducts/data.data'
inputTestFile = "../data/jester-data-1/50users/data_test.data"
class process_reduct:
    def __init__(self,train_set, user_reduct):
        self.train_set = train_set
        self.user_reduct = user_reduct
        self.dic_reducts = {}
        self.dict_fitting_finding = {}
        self.count = 0
        self.cl = {}
        self.dl = {}
        self.len_item = 20
        # tinh dl tren dan test
        self.train_set_test = ReadFile(inputTestFile).read()
    def process_finding(self):
            self.reduct_finding_user(self.user_reduct)
            self.process_fitting_finding(self.user_reduct)
            return self.dic_reducts

    def reduct_finding_user(self, user):
        print("-------------Reduct_Finding---------------")
        cl = list(set(self.train_set["items_seen_by_user"][user]))
        dl = list(set(self.train_set_test["items_seen_by_user"][user]))
        # dl = list(set(self.train_set["items"]) - set(cl))
        for item in self.train_set["items_unobserved"]:
            if(len(dl) < 20 and item not in dl):
                dl.append(item)
            elif(len(dl) > 20):
                break
        # Lay ngau nhien 20 bo phim chua danh gia lam dan dieu kien
        self.cl = cl = set(random.sample(cl, k=int(self.len_item)))
        self.dl = dl = set(random.sample(dl, k=int(self.len_item)))
        # self.dl = set(dl)

        top = []
        # for i in range(len(cl)):
        #     top += itertools.combinations(cl, i + 1)
        # self.top = len(top)
        self.top = pow(2, len(cl)) - 1
        self.dic_reducts.setdefault(user, {'cl' : [cl]})
        # Tinh do phu thuoc cua CL vao DL
        dependencyCL = self.calculateDependency(cl, dl)

        # phan chinh thuat toan reduct_finding
        self.reduct_finding(cl, cl, user, dl, dependencyCL)

    # Ham tinh do phu thuoc
    def calculateDependency(self, cl, dl):
        covCLu = {}
        covDLu = {}
        topCL = []

        # cover_cl = self.calculateCoverS(cl)
        topCL = self.calculateCoverS(cl)
        topDL = self.calculateCoverS(dl)

        # for i in range(len(cl)):
        #     top += itertools.combinations(cl, i + 1)
        # self.top = len(top)

        # for each user tìm covClu và covDLu dựa trên cover_cl và cover_dl
        for user in self.train_set["users"]:
            covCLu.setdefault(user, set.intersection(*[set(x) for x in topCL if user in x]))
            covDLu.setdefault(user, set.intersection(*[set(x) for x in topDL if user in x]))

        posCL = []
        for (key, value) in covCLu.items():
            for (keyDL, valueDL) in covDLu.items():
                join = set(value) & set(valueDL)
                if join not in posCL and join != set():
                    posCL.append(join)

        pCL = len(posCL) / self.top

        return pCL


    # Ham tinh dan phu CL, DL
    def calculateCoverS(self, set_items):
        # coverS = []
        # topCl = []
        # for i in range(0, len(set_items)):
        #     combination = list(itertools.combinations(set_items, i + 1))
        #     for element in combination:
        #         cover = [user for (user, items) in self.train_set["items_seen_by_user"].items()
        #                  if
        #                  set(element).issubset(items)]
        #         if cover not in coverS and len(cover) > 0:
        #             topCl.append(cover)
        #         coverS.append(cover)
        # # Neu User khong o trong phu dinh
        # if self.train_set["users"] not in topCl:
        #     topCl.append(self.train_set["users"])

        coverS = []
        topItem = []

        # Phu dinh bang phu cua tap muc lon nhat
        for item in set_items:
            cover = [user for (user, items) in self.train_set["items_seen_by_user"].items()
                     if
                     item in list(items)]
            if cover not in coverS and len(cover) > 0:
                topItem.append(cover)
            coverS.append(cover)

        #  Them tap U vao top
        if self.train_set["users"] not in topItem:
            topItem.append(self.train_set["users"])
        return topItem

    # Hàm sinh GCRL
    # ccl là dàn điều kiện hiện thời
    # pccl là dàn cha
    def reduct_finding(self, ccl, pccl, user, dl, dependencyCL):
        # Lay 1 phu trong dan con
        if len(self.dic_reducts[user]['cl'][0]) < self.len_item:
            return
        self.count += 1
        sccl = self.generateAllChid(ccl)
        if ccl == pccl:
            for item in sccl:
                self.reduct_finding(item, ccl, user, dl, dependencyCL)
        else:
            dependencyCCL = self.calculateDependency(ccl, dl)

            if dependencyCCL == dependencyCL:
                if(ccl not in self.dic_reducts[user]['cl']):
                    self.dic_reducts[user]['cl'].append(ccl)
                if pccl in self.dic_reducts[user]['cl']:
                    self.dic_reducts[user]['cl'].remove(pccl)

                for item in sccl:
                    self.reduct_finding(item, ccl, user, dl, dependencyCL)
        print(self.count)
        print(self.dic_reducts)



 # Hàm sinh ra tất cả các tập con từ một tập cha
    def generateAllChid(self, parent):
        childs = []
        for i in range(len(parent) - 1):
            childs += itertools.combinations(parent, i + 1)

        return childs


#Bắt đầu thuật toán fitting_finding
    def process_fitting_finding(self, user):
        print("-------------Fitting_Finding---------------")
        self.fitting_finding_user(user)

    def fitting_finding_user(self, user):
        cl = self.dic_reducts.get(user).get('cl')[0]
        dl = self.dl
        self.dic_reducts[user]['dl'] = [dl]
        childsDL = self.generateAllChid(dl)

        dependency = self.calculateDependency(cl, dl)
        for child in childsDL:
            # Lay 1 thich nghi
            if len(self.dic_reducts[user]['dl']) == 2:
                self.dic_reducts[user]['dl'].pop(0)
                break
            self.fitting_finding(cl, child, dependency, user)

    def fitting_finding(self, cl, cdl, dependency, user):
        self.count += 1
        dependencyCDL = self.calculateDependency(cl, cdl)

        if dependencyCDL >= dependency:
            self.dic_reducts[user]['dl'].append(cdl)
            # return
        # else:
        #     childs = self.generateAllChid(cdl)
        #
        #     for child in childs:
        #         self.fitting_finding(cl, child, dependency, user)
        print(self.count)
        print(self.dic_reducts)