class MLM_learn():
    def __init__(self, alpha=1, train_set_cl=None, train_set_dl=None):
        self.alpha = alpha
        self.train_set_cl = train_set_cl
        self.train_set_dl = train_set_dl
        self.cl = self.train_set_cl['items_seen_by_user'][1]
        self.dl = self.train_set_dl['items_seen_by_user'][1]
        self.lis_user_covCl = set()
        self.N_covCl = {}
        self.N_covDl = {}
        self.neighbors_user = {}
        self.Label_user = {}
        self.l_user = {}

    # Ham tinh phu dinh
    def calculateTopCoverS(self, set_items, train_set):
        coverS = []
        topItem = []
        # Phu dinh bang phu cua tap muc lon nhat

        for item in set_items:
            cover = [user for (user, items) in train_set["items_seen_by_user"].items()
                     if
                     item in list(items)]
            if cover not in coverS and len(cover) > 0:
                topItem.append(cover)
            coverS.append(cover)

        #  Them tap U vao top
        if train_set["users"] not in topItem:
            topItem.append(train_set["users"])

        return topItem

    # Ham tim phu cam sinh
    def calculateCoverS(self, set_items, train_set):
        cov_user = {}
        CoverItem = []

        # for each user tìm covClu và covDLu dựa trên cover_cl và cover_dl
        for user in train_set["users"]:
            cov_user.setdefault(user, set.intersection(*[set(x) for x in set_items if user in x]))

        for user in cov_user:
            if list(cov_user[user]) not in CoverItem:
                CoverItem.append(list(cov_user[user]))
        return CoverItem

    # Ham tinh do tin cay
    def caculatorReliability(self, lu, user):
        user_contains_lu = []
        for u in self.neighbors_user[user]:
            if lu in self.train_set_dl["items_seen_by_user"][u]:
                user_contains_lu.append(u)
        reliability = len(user_contains_lu) / len(self.neighbors_user[user])
        return reliability

    def MLM_learn(self):
        # Tim phu dinh cua CL va DL

        topCL = self.calculateTopCoverS(self.cl, self.train_set_cl)
        topDL = self.calculateTopCoverS(self.dl, self.train_set_dl)

        # Tinh phu cam sinh CL va DL
        CovCl = self.calculateCoverS(topCL, self.train_set_cl)
        CovDl = self.calculateCoverS(topDL, self.train_set_dl)

        for user in self.train_set_cl["users"]:
            self.N_covCl.setdefault(user, self.train_set_cl["users"])
            self.N_covDl.setdefault(user, self.train_set_cl["users"])
            self.neighbors_user.setdefault(user, {user})
            self.Label_user[user] = set()
            self.l_user[user] = set()

            # Buoc 4 thuat toan
            for cov_cl in CovCl:
                if user in cov_cl:
                    self.N_covCl[user] = (self.N_covCl[user] & set(cov_cl))

            for cov_dl in CovDl:
                if user in cov_dl:
                    self.N_covDl[user] = (self.N_covDl[user] & set(cov_dl))
            # Buoc 5 thuat toan
            # Tim dan lang gieng gan cua user trong CovCl va CovDl

            # self.neighbors_user[user].update(self.N_covCl[user])
            self.neighbors_user[user].update(self.N_covCl[user] & self.N_covDl[user])
            # self.neighbors_user[user].update(self.N_covDl[user])

            # Buoc 6 thuat toan
            for u_neighbors in self.neighbors_user[user]:
                item_seem_by_cov = self.train_set_dl["items_seen_by_user"][u_neighbors]
                self.Label_user[user].update(item_seem_by_cov)

            # Buoc 7 thuat toan
            for lu in self.Label_user[user]:
                if (self.caculatorReliability(lu, user) >= self.alpha):
                    self.l_user[user].update(set([lu]))
        # End for tra ve mo hinh phan lop
        return self.l_user
