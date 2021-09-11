import csv

import numpy as np
import xlrd
from joblib.numpy_pickle_utils import xrange
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import xlsxwriter
from numpy import asarray
from numpy import savetxt


class processData():
    def __init__(self, out_reducts_file=None, data=None, input_cl_file=None, input_dl_file=None, \
                 input_test_file=None, sep=' ', num_topics=10, num_doc=1350, mode_write=None):
        self.out_file = out_reducts_file
        self.mode = 'w'
        self.sep = sep
        self.data = data
        self.check = 0
        self.input_cl_file = input_cl_file
        self.input_dl_file = input_dl_file
        self.sep = sep
        self.doccument = 0
        self.items = num_topics
        self.users = num_doc
        self.matrix = np.zeros((self.users, self.items))
        self.matrix_test = None
        self.input_test_file = input_test_file
        self.mode_write = mode_write

    # Ham doc file .theta
    def readFileCl(self):
        list_users = set()
        list_items = set()
        dict_feedback = {}
        items_seen_by_user = {}

        with open(self.input_file) as infile:

            for line in infile:
                self.doccument += 1
                list_users.add(self.doccument)

                if line.strip():
                    inline = line.split(self.sep)
                    for idx, val in enumerate(inline):
                        if (idx < self.items):
                            self.matrix[self.doccument - 1][idx] = float(val)

        matrix_none_avg = list(self.matrix)

        for i in range(1, self.items + 1):
            list_items.add(i)

        average = np.zeros(self.items)

        # Xu ly du lieu. Chi lay nhung chu de co gia tri lon hon trung binh
        for j in range(self.items):
            avg = float(0)
            for i in range(self.users):
                avg += self.matrix[i][j]

            average[j] = avg / self.users

        # Neu nhu gia tri xac suat be hon gia tri trung binh thi gan = 0
        for j in range(self.items):
            for i in range(self.users):
                if (self.matrix[i][j] < average[j]):
                    self.matrix[i][j] = 0

        for i in range(self.users):
            for j in range(self.items):
                if (self.matrix[i][j] != 0):
                    items_seen_by_user.setdefault(i + 1, set()).add(j)
                    dict_feedback.setdefault(i + 1, {}).update({j: self.matrix[i][j]})

        dict_file = {
            'feedback': dict_feedback,
            'users': list_users,
            'items': list_items,
            'items_seen_by_user': items_seen_by_user,
            'matrix_with_avg': self.matrix,
            'matrix_none_avg': matrix_none_avg
        }

        return dict_file

    def readFilleDl(self):

        x = xlrd.open_workbook(self.input_dl_file)
        x1 = x.sheet_by_name('labels')

        list_labels = set([0, 1, 2, 3, 4])
        list_users = set()
        labels_seen_by_user = {}

        for rownum in xrange(x1.nrows):
            #
            list_users.add(rownum + 1)
            for idx, val in enumerate(x1.row_values(rownum)):
                if (type(val) is not str and val != ''):
                    if (val > 0):
                        labels_seen_by_user.setdefault(rownum + 1, set()).add(idx)
        dict_file = {
            'users': list_users,
            'items': list_labels,
            'items_seen_by_user': labels_seen_by_user
        }

        return dict_file

    # Tra ve ma tran test va train tinh do tuong tu
    def readFileDataTest(self):
        # Ma tran user test
        self.matrix_test = np.zeros((150, self.items))
        if (self.input_test_file != None):
            with open(self.input_test_file) as infile:
                for line in infile:
                    self.doccument += 1
                    if line.strip():
                        inline = line.split(self.sep)
                        for idx, val in enumerate(inline):
                            if (idx < self.items):
                                self.matrix_test[self.doccument - 1][idx] = float(val)

        # Ma tran user train
        self.doccument = 0
        self.matrix = np.zeros((self.users, self.items))
        if (self.input_file != None):
            with open(self.input_file) as infile:
                for line in infile:
                    self.doccument += 1
                    if line.strip():
                        inline = line.split(self.sep)
                        for idx, val in enumerate(inline):
                            if (idx < self.items):
                                self.matrix[self.doccument - 1][idx] = float(val)
        dict_file = {
            'matrix_train': self.matrix,
            'matrix_test': self.matrix_test
        }
        return dict_file

    # Ham ghi ket qua
    def WriteFile(self):
        with open(self.out_file, self.mode) as infile:
            if (self.mode_write == 'class_train'):
                for user in self.data:
                    infile.write('%d%s' % (user, " : "))
                    for label in self.data[user]:
                        infile.write('%d%s' % (label, "\t"))
                    infile.write("\n")
            elif (self.mode_write == 'class_test'):
                for user_test in self.data:
                    infile.write('%d%s' % (user_test, " : "))
                    for user_train in self.data[user_test]:
                        infile.write('%d%s' % (user_train, " : "))
                        for label in self.data[user_test][user_train]:
                            infile.write('%d%s' % (label, "\t"))
                    infile.write("\n")

    def WriteClassi(self):
        with open(self.out_file, self.mode) as infile:

            for user, val in enumerate(self.data):
                infile.write('%d%s' % (user + 1, " : "))
                for label, val_label in enumerate(val):
                    if (val_label == 1):
                        infile.write('%d%s' % (label, "\t"))
                infile.write("\n")

    # Ham tinh tf_idf
    def readDocument(self, type):
        list_users = set()
        list_items = set()
        items_seen_by_user = {}

        with open(self.input_cl_file, "r", encoding="utf8") as sentences_file:
            reader = csv.reader(sentences_file, delimiter='\n')
            next(reader)
            corpus = []
            for row in reader:
                corpus += row
        cont_vect = CountVectorizer()
        self.create_document_term_matrix(corpus, cont_vect)

        tf_idf_vect = TfidfVectorizer()
        vector_tf_idf = self.create_document_term_matrix(corpus, tf_idf_vect).to_numpy()

        # Chia 1350 train, 150 test voi du lieu test 1 va 10
        training, test = vector_tf_idf[0:1350, ], vector_tf_idf[1350:1500, :]

        # Chia 1350 train, 150 test voi du lieu test 2->9
        # training1, test = vector_tf_idf[0:150, ], vector_tf_idf[150:300, :]
        # training2 = vector_tf_idf[300:1500, :]
        # training = np.concatenate((training1, training2), axis=0)

        # # Them LDA vao du lieu
        # average_training = np.zeros(self.items)
        # average_test = np.zeros(self.items)
        # self.input_test_file = '../../data/Classification/10ChuDe/1/data.test.theta'
        # self.input_file = '../../data/Classification/10ChuDe/1/data.dat.theta'
        # matrix_lda = self.readFileDataTest()
        # # Du lieu train va test
        # training = np.concatenate((matrix_lda['matrix_train'], training), axis=1)
        # test = np.concatenate((matrix_lda['matrix_test'], test), axis=1)
        #
        # # Xu ly du lieu. Chi lay nhung chu de co gia tri lon hon trung binh
        # for j in range(self.items):
        #     avg_train = avg_test = float(0)
        #     for i in range(self.users):
        #         avg_train += training[i][j]
        #     average_training[j] = avg_train / self.users
        #
        #
        #     for i in range(150):
        #         avg_test += test[i][j]
        #     average_test[j] = avg_test / self.users
        # # Neu nhu gia tri xac suat be hon gia tri trung binh thi gan = 0
        # for j in range(self.items):
        #     for i in range(self.users):
        #         if (training[i][j] < average_training[j]):
        #             training[i][j] = 0

        # Doc file TF_IDF chi Huyen
        # with open(self.input_cl_file, "r", encoding="utf8") as sentences_file:
        #     reader = csv.reader(sentences_file, delimiter='\n')
        #     corpus = []
        #     for row in reader:
        #         data = row[0].split(",")
        #         corpus.append(data)
        #
        #
        # vector_tf_idf = numpy.array(corpus)
        # vector_tf_idf = vector_tf_idf.astype(np.float)
        #
        # # Chia 1350 train, 150 test
        # training1, test = vector_tf_idf[0:150, :], vector_tf_idf[150:300, :]
        # training2 = vector_tf_idf[300:1500, :]
        # training =  np.concatenate((training1, training2), axis = 0)
        # End

        if (type == 'train'):
            matrix = training
            users = training.shape[0]
            items = training.shape[1]

            for i in range(1, users + 1):
                list_users.add(i)
            for i in range(1, items + 1):
                list_items.add(i)

            for i in range(users):
                for j in range(items):
                    if (training[i][j] != 0):
                        items_seen_by_user.setdefault(i + 1, set()).add(j)
        elif (type == 'test'):
            matrix = test
            users = test.shape[0]
            items = test.shape[1]

            for i in range(1, users + 1):
                list_users.add(i)
            for i in range(1, items + 1):
                list_items.add(i)

            for i in range(users):
                for j in range(items):
                    if (test[i][j] != 0):
                        items_seen_by_user.setdefault(i + 1, set()).add(j)
        dict_file = {
            'users': list_users,
            'items': list_items,
            'items_seen_by_user': items_seen_by_user,
            'matrix_tf_idf': matrix
        }

        return dict_file

    # Ham tao ma tran tf_idf
    def create_document_term_matrix(self, review_list, vectorizer):
        doc_term_matrix = vectorizer.fit_transform(review_list)
        return DataFrame(doc_term_matrix.toarray(),
                         columns=vectorizer.get_feature_names())

    # Data be 17/07/21
    # Ham tinh tf_idf
    def readDocumentMin(self, type):
        list_users = set()
        list_items = set()
        items_seen_by_user = {}

        with open(self.input_cl_file, "r", encoding="utf8") as sentences_file:
            reader = csv.reader(sentences_file, delimiter='\n')
            next(reader)
            corpus = []
            for row in reader:
                corpus += row
        cont_vect = CountVectorizer()
        self.create_document_term_matrix(corpus, cont_vect)

        tf_idf_vect = TfidfVectorizer()
        vector_tf_idf = self.create_document_term_matrix(corpus, tf_idf_vect).to_numpy()

        # Chia 1350 train, 150 test
        training, test = vector_tf_idf[0:10, :], vector_tf_idf[10:15, :]

        if (type == 'train'):
            matrix = training
            savetxt("../DATA/input/tfidf_train.xlsx", matrix, delimiter=',')
            users = training.shape[0]
            items = training.shape[1]

            for i in range(1, users + 1):
                list_users.add(i)
            for i in range(1, items + 1):
                list_items.add(i)

            for i in range(users):
                for j in range(items):
                    if (training[i][j] != 0):
                        items_seen_by_user.setdefault(i + 1, set()).add(j)
        elif (type == 'test'):
            matrix = test
            savetxt("../DATA/input/tfidf_test.xlsx", matrix, delimiter=',')
            users = test.shape[0]
            items = test.shape[1]

            for i in range(1, users + 1):
                list_users.add(i)
            for i in range(1, items + 1):
                list_items.add(i)

            for i in range(users):
                for j in range(items):
                    if (test[i][j] != 0):
                        items_seen_by_user.setdefault(i + 1, set()).add(j)
        dict_file = {
            'users': list_users,
            'items': list_items,
            'items_seen_by_user': items_seen_by_user,
            'matrix_tf_idf': matrix
        }

        return dict_file

    def readFilleDlMin(self):

        x = xlrd.open_workbook(self.input_dl_file)
        x1 = x.sheet_by_name('labels')

        list_labels = set([0, 1, 2, 3, 4])
        list_users = set()
        labels_seen_by_user = {}

        for rownum in xrange(x1.nrows):
            # Comment tao data be hon
            if (rownum < 10):
                #
                list_users.add(rownum + 1)
                for idx, val in enumerate(x1.row_values(rownum)):
                    if (type(val) is not str and val != ''):
                        if (val > 0):
                            labels_seen_by_user.setdefault(rownum + 1, set()).add(idx)
        dict_file = {
            'users': list_users,
            'items': list_labels,
            'items_seen_by_user': labels_seen_by_user
        }

        return dict_file
