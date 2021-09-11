import math

from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.utils.process_data import ReadFile




class recommenderSystem(UserKNN):
    def __init__(self, inputTrainFile=None, test_file=None, out_resultFile=None, reducts=None, user_recommender=None,
                 k_neighbors=5):
        self.inputTrainFile = inputTrainFile
        self.test_file = test_file
        self.out_resultFile = out_resultFile
        self.k_neighbors = k_neighbors
        self.train_sets = ReadFile(self.inputTrainFile).read()
        self.reducts = reducts
        self.simU_V = {}
        self.u_recomend = user_recommender


    # Ham tim nguoi dung tuong tu
    def find_neighbors_user(self):
        k_neighbors_user = {}
        for u in self.train_sets['users']:
            numerator_sim = pu_i = pv_i = float(0)
            # for item in self.train_sets['items_seen_by_user'][u]:
            for item in self.reducts[self.u_recomend]['cl'][0]:
                if item in self.train_sets['items_seen_by_user'][u]:
                    numerator_sim += (
                                self.train_sets['feedback'][u][item] * self.train_sets['feedback'][self.u_recomend][
                            item])
                    pu_i += pow(self.train_sets['feedback'][u][item], 2)
                    pv_i += pow(self.train_sets['feedback'][self.u_recomend][item], 2)

            denominator_sim = math.sqrt(pu_i * pv_i)
            sim = float(
                (numerator_sim if numerator_sim != 0 else -1) / (denominator_sim if denominator_sim != 0 else 1))

            self.simU_V.setdefault(u, sim)
        # Sap xep lai gia tri giam dan
        self.simU_V = dict(sorted(self.simU_V.items(), key=lambda item: item[1], reverse=True))

        i = 0
        for u in self.simU_V:
            if i < self.k_neighbors and u != self.u_recomend:
                k_neighbors_user.setdefault(u, self.simU_V[u])
                i += 1
        return k_neighbors_user

    # Ham tinh do quan tam
    def levelConcern(self, k_neighbors_user):
        self.levelConcern = {}
        for item in self.reducts[self.u_recomend]['dl'][0]:
            pU_I = denominator = numerator = 0
            for u_neighbors in k_neighbors_user:
                # chi lay nhung gia tri ma nguoi dung u_neighbors co danh gia
                if item in self.train_sets['feedback'][u_neighbors]:
                    numerator += k_neighbors_user[u_neighbors] * \
                                 (self.train_sets['feedback'][u_neighbors][item] - self.avg_ratinng(u_neighbors,
                                                                                                    self.train_sets))

                denominator += k_neighbors_user[u_neighbors]
            pU_I = self.avg_ratinng(self.u_recomend, self.train_sets) + numerator / denominator
            self.levelConcern.setdefault(item, pU_I)

        self.levelConcern = dict(sorted(self.levelConcern.items(), key=lambda item: item[1], reverse=True))
        return self.levelConcern

    # Ham tu van
    def recommender(self):
        k_neighbors_user = self.find_neighbors_user()
        recommender = self.levelConcern(k_neighbors_user)
        return recommender

    # Ham tinh do do cosin
    def cosineSimilarity(self, X, Y):
        numeator = sum(a * b for a, b in zip(X, Y))
        denominator = self.square_rooted(X) * self.square_rooted(Y)
        return numeator / float(denominator)

    def square_rooted(self, x):
        return round(math.sqrt(sum([a * a for a in x])), 5)

    # Tinh muc danh gia trung binh cua nguoi dung
    def avg_ratinng(self, u, train_sets):
        avg = float(0)
        for item in train_sets['items_seen_by_user'][u]:
            avg += train_sets['feedback'][u][item]
        return float(avg / len(train_sets['items_seen_by_user'][u]))
