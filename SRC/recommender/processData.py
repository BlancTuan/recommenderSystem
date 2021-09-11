import random

import xlrd
from caserec.utils.process_data import ReadFile
from caserec.utils.process_data import WriteFile
from joblib.numpy_pickle_utils import xrange

out_train_file = "../../data/jester-data-1/50users/data_train.data"
out_test_file = "../../data/jester-data-1/50users/data_test.data"
out_file = "../../data/ml-100k/u.data"


class convertFile():
    def __init__(self, out_reducts_file=None, data=None, input_file=None, sep=' ', num_topics=10, num_doc=1500):
        self.out_file = out_reducts_file
        self.mode = 'w'
        self.sep = sep
        self.data = data
        self.check = 0
        self.input_file = "../../data/Classification/10ChuDe/data.dat.theta"
        self.input_dl_file = "../../data/Classification/10ChuDe/data_full.xlsx"
        self.sepace = '\t'
        self.doccument = 0
        self.items = num_topics
        self.users = num_doc
        self.seperate = 0.2
        # self.matrix = np.zeros((self.users, self.items))

    def xls_to_csv(self):
        num_user_recommen = 0

        x = xlrd.open_workbook('../../data/jester-data-1/jester-data-1.xls')
        x1 = x.sheet_by_name('jester-data-1-new')

        list_item_seem_by_user_test = []
        list_item_seem_by_user_train = []
        for rownum in xrange(x1.nrows):  # To determine the total rows.
            self.train = 0
            for idx, val in enumerate(x1.row_values(rownum)):
                # Lay nhung user co rating tren 50 phim lam train
                if idx == 0 and val >= 50:
                    self.train = 1
                # Lấy so user tu idx = 1 va co danh gia (!=99)
                if idx != 0 and val != 99 and self.train == 1:
                    list_item_seem_by_user_train.append([rownum, idx, val])
                elif idx != 0 and val != 99 and num_user_recommen < 50:
                    list_item_seem_by_user_test.append([rownum, idx, val])

            if self.train == 0:
                num_user_recommen += 1
        WriteFile(out_file, list_item_seem_by_user_test).write()
        # WriteFile(out_train_file,list_item_seem_by_user_train).write()

    def WriteFile(self):
        with open(self.out_file, self.mode) as infile:
            for user in self.data:
                for pair in self.data[user]['cl']:
                    infile.write('%d%s' % (user, " CL:"))
                    for item in pair:
                        infile.write('%s%d' % (self.sep, item))
                    infile.write("\n")
                for pair in self.data[user]['dl']:
                    infile.write('%d%s' % (user, " DL:"))
                    for item in pair:
                        infile.write('%s%d' % (self.sep, item))
                    infile.write("\n")

    def WriteResult(self):
        with open(self.out_file, self.mode) as infile:
            for u in self.data:
                for item in self.data[u]:
                    infile.write('%d%s%d%s%f\n' % (u, self.sepace, item, self.sepace, self.data[u][item]))

    def processData(self):
        transet = ReadFile(out_file).read()
        train_data = []
        test_data = []
        num_user = 0
        # lay ngau nhien 20% danh gia cua nguoi dung u
        for u in transet['users']:
            if num_user < 50 and len(transet["feedback"][u]) >= 25:
                num_user += 1
                # Chọn ngẫu nhiên 20% mục đã đánh giá lam du lieu test
                test_item = random.choices(list(transet["items_seen_by_user"][u]),
                                           k=int(len(transet["items_seen_by_user"][u]) * self.seperate))

                for i in transet['feedback'][u]:
                    if i in test_item:
                        test_data.append([u, i, transet["feedback"][u][i]])
                    else:
                        train_data.append([u, i, transet["feedback"][u][i]])

        WriteFile(out_train_file, train_data).write()
        WriteFile(out_test_file, test_data).write()
        return 1

# convertFile = convertFile()
# convertFile.processData()
# convertFile.xls_to_csv()
# convertFile.ReadFile()
# convertFile.readDlFille()
