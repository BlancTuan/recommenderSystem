import time

from MLM_learn import MLM_learn
from Process_Data import processData
from SRC.MLM_classifier import MLM_classifier

start_time = time.time()

input_cl_file = "../DATA/input/data_test.csv"

# # 150 review thu 1
# input_dl_file = "../DATA/input/1/data_dat.xlsx"
# input_file_test = "../DATA/input/1/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/1/classification_train.dat"
# out_class_test_file = "../DATA/output/1/classification_test.dat"

# # 150 review thu 2
# input_dl_file = "../DATA/input/2/data_dat.xlsx"
# input_file_test = "../DATA/input/2/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/2/classification_train.dat"
# out_class_test_file = "../DATA/output/2/classification_test.dat"
#
# # 150 review thu 3
# input_dl_file = "../DATA/input/3/data_dat.xlsx"
# input_file_test = "../DATA/input/3/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/3/classification_train.dat"
# out_class_test_file = "../DATA/output/3/classification_test.dat"
#
# 150 review thu 4
# input_dl_file = "../DATA/input/4/data_dat.xlsx"
# input_file_test = "../DATA/input/4/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/4/classification_train.dat"
# out_class_test_file = "../DATA/output/4/classification_test.dat"
#
# # 150 review thu 5
# input_dl_file = "../DATA/input/5/data_dat.xlsx"
# input_file_test = "../DATA/input/5/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/5/classification_train.dat"
# out_class_test_file = "../DATA/output/5/classification_test.dat"
#
# # 150 review thu 6
# input_dl_file = "../DATA/input/6/data_dat.xlsx"
# input_file_test = "../DATA/input/6/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/6/classification_train.dat"
# out_class_test_file = "../DATA/output/6/classification_test.dat"
#
# # 150 review thu 7
# input_dl_file = "../DATA/input/7/data_dat.xlsx"
# input_file_test = "../DATA/input/7/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/7/classification_train.dat"
# out_class_test_file = "../DATA/output/7/classification_test.dat"
#
# # 150 review thu 8
# input_dl_file = "../DATA/input/8/data_dat.xlsx"
# input_file_test = "../DATA/input/8/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/8/classification_train.dat"
# out_class_test_file = "../DATA/output/8/classification_test.dat"
#
# 150 review thu 9
# input_dl_file = "../DATA/input/9/data_dat.xlsx"
# input_file_test = "../DATA/input/9/data_test.xlsx"
# # Out file
# out_class_train_file = "../DATA/output/9/classification_train.dat"
# out_class_test_file = "../DATA/output/9/classification_test.dat"
#
# 150 review thu 10
input_dl_file = "../DATA/input/10/data_dat.xlsx"
input_file_test = "../DATA/input/10/data_test.xlsx"
# Out file
out_class_train_file = "../DATA/output/10/classification_train.dat"
out_class_test_file = "../DATA/output/10/classification_test.dat"


# # Data be
# input_cl_min_file = "../DATA/input/data_test.csv"
class classification():
    def __init__(self):
        self.input_cl_file = input_cl_file
        self.input_dl_file = input_dl_file
        self.out_class_train_file = out_class_train_file
        self.out_class_test_file = out_class_test_file
        self.train_set_cl_train = processData(input_cl_file=self.input_cl_file).readDocument(type='train')
        self.train_set_cl_test = processData(input_cl_file=self.input_cl_file).readDocument(type='test')
        self.train_set_dl_train = processData(input_dl_file=input_dl_file).readFilleDl()

    def result(self):
        # Thuat toan phan lop
        Classification = MLM_learn(train_set_cl=self.train_set_cl_train, alpha=1,
                                   train_set_dl=self.train_set_dl_train).MLM_learn()

        classi = MLM_classifier(Classification, 1,
                                train_set_train=self.train_set_cl_train,
                                train_set_test=self.train_set_cl_test).MLM_classifier()

        # Danh gia mo hinh
        MLM_classifier.evalue(input_file_test=input_file_test, classi=classi)
        # Ghi lai ket qua
        processData(self.out_class_train_file, data=Classification, mode_write='class_train').WriteFile()
        # processData(self.out_class_test_file, data=classi, mode_write='class_test').WriteFile()
        processData(self.out_class_test_file, data=classi).WriteClassi()


classification = classification()
classification.result()
