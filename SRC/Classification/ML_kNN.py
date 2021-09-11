import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from Process_Data import processData

input_cl_file = "../DATA/input/data_test.csv"

# # 150 review thu 1
# input_dl_file = "../DATA/input/1/data_dat.xlsx"
# input_file_test = "../DATA/input/1/data_test.xlsx"

# # 150 review thu 2
# input_dl_file = "../DATA/input/2/data_dat.xlsx"
# input_file_test = "../DATA/input/2/data_test.xlsx"

# # 150 review thu 3
# input_dl_file = "../DATA/input/3/data_dat.xlsx"
# input_file_test = "../DATA/input/3/data_test.xlsx"

# 150 review thu 4
# input_dl_file = "../DATA/input/4/data_dat.xlsx"
# input_file_test = "../DATA/input/4/data_test.xlsx"

# # 150 review thu 5
# input_dl_file = "../DATA/input/5/data_dat.xlsx"
# input_file_test = "../DATA/input/5/data_test.xlsx"

# # 150 review thu 6
# input_dl_file = "../DATA/input/6/data_dat.xlsx"
# input_file_test = "../DATA/input/6/data_test.xlsx"

# # 150 review thu 7
# input_dl_file = "../DATA/input/7/data_dat.xlsx"
# input_file_test = "../DATA/input/7/data_test.xlsx"

# # 150 review thu 8
# input_dl_file = "../DATA/input/8/data_dat.xlsx"
# input_file_test = "../DATA/input/8/data_test.xlsx"

# 150 review thu 9
# input_dl_file = "../DATA/input/9/data_dat.xlsx"
# input_file_test = "../DATA/input/9/data_test.xlsx"

# 150 review thu 10
input_dl_file = "../DATA/input/10/data_dat.xlsx"
input_file_test = "../DATA/input/10/data_test.xlsx"

train_set_train = processData(input_cl_file=input_cl_file).readDocument(type='train')
train_set_test = processData(input_cl_file=input_cl_file).readDocument(type='test')

train_set_dl = processData(input_dl_file=input_dl_file).readFilleDl()
train_set = processData(input_dl_file=input_file_test).readFilleDl()

X_train = train_set_train['matrix_tf_idf']
y_train = np.zeros((len(train_set_dl['users']), 5))
X_test = train_set_test['matrix_tf_idf']
y_test = np.zeros((len(train_set['users']), 5))

for u in train_set_dl['items_seen_by_user']:
    for labels in train_set_dl['items_seen_by_user'][u]:
        y_train[u - 1][labels] = 1

for u in train_set['items_seen_by_user']:
    for labels in train_set['items_seen_by_user'][u]:
        y_test[u - 1][labels] = 1

# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=10)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
predictions = knn.predict(X_test)

# print(predictions)
# Danh gia mo hinh
recal = metrics.classification_report(y_test, predictions)
hamming_loss = metrics.hamming_loss(y_test, predictions)
zero_one_loss = metrics.zero_one_loss(y_test, predictions)
coverage_error = metrics.coverage_error(y_test, predictions)
label_ranking_loss = metrics.label_ranking_loss(y_test, predictions)
average_precision_score = metrics.average_precision_score(y_test, predictions)
print(recal)
print('----------------------------------------')
print('hamming_los:', hamming_loss)
print('zero_one_loss:', zero_one_loss)
print('coverage_error:', coverage_error)
print('label_ranking_loss:', label_ranking_loss)
print('average_precision_score:', average_precision_score)
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
