import matplotlib.pyplot as plt

predictions_file = '../../data/export/recommend/recommen.data'
reduct_file = '../../data/export/recommend/reduct.data'
test_file = '../../data/jester-data-1/50users/data_test.data'


# Lop danh gia mo hinh
class modelEvaluation():
    def __init__(self):
        self.reduct_data = 1
        self.num_user = 50

    def evalue(self):
        sum_cl = sum_dl = 0
        MAE = 0
        RMSE = 0
        train_set = self.readReduct()
        for u in train_set['ratio']:
            sum_cl += train_set['ratio'][u]['CL:']
            sum_dl += train_set['ratio'][u]['DL:']
        reduction_rate_CL = sum_cl / self.num_user
        reduction_rate_DL = sum_dl / self.num_user

        for user in train_set['dict_feedback_pre']:
            len_recomen = 0
            MAE_recommen = 0
            RMSE_recommen = 0
            for item in train_set['dict_feedback_pre'][user]:
                if item in train_set['dict_feedback_test'][user]:
                    len_recomen += 1
                    MAE_recommen += abs(
                        train_set['dict_feedback_pre'][user][item] - train_set['dict_feedback_test'][user][item])
                    RMSE_recommen += pow(
                        (train_set['dict_feedback_pre'][user][item] - train_set['dict_feedback_test'][user][item]), 2)

            RMSE += RMSE_recommen / (len_recomen if len_recomen > 0 else 1)
            MAE += MAE_recommen / (len_recomen if len_recomen > 0 else 1)
        MAE = MAE / self.num_user
        RMSE = RMSE / self.num_user

        # Ve bieu do.
        ratio_cl = []
        ratio_dl = []
        user = []

        for u in train_set['ratio']:
            user.append(u)
            ratio_cl.append(train_set['ratio'][u]['CL:'])
            ratio_dl.append(train_set['ratio'][u]['DL:'])
        plt.figure(figsize=(12, 5), dpi=100)
        plt.xlabel('User_id')
        plt.ylabel('Ratio')
        plt.plot(user, ratio_cl, label='CL')
        plt.plot(user, ratio_dl, label='DL')
        plt.legend(loc='best')
        # plt.title("Check")
        plt.show()
        print('Tỷ lệ giảm thiểu phủ trung bình:')
        print('CL : ', reduction_rate_CL)
        print('DL : ', reduction_rate_DL)

        print('Tỷ lệ recommender vao muc đánh giá test: ')
        print('MAE : ', MAE)
        print('RMSE: ', RMSE)

    def readReduct(self):
        dict_feedback_pre = {}
        dict_feedback_test = {}
        result = {}
        with open(reduct_file) as infile:
            for line in infile:
                reduct = line.split()
                ratio = 1 - ((len(reduct) - 2) / 20)
                result.setdefault(reduct[0], {}).setdefault(reduct[1], ratio)

        with open(predictions_file) as infile:
            for line in infile:
                inline = line.split('\t')
                user, item, value = int(inline[0]), int(inline[1]), float(inline[2])
                dict_feedback_pre.setdefault(user, {}).update({item: value})

        with open(test_file) as infile:
            for line in infile:
                inline = line.split('\t')
                user, item, value = int(inline[0]), int(inline[1]), float(inline[2])
                dict_feedback_test.setdefault(user, {}).update({item: value})
        train_set = {
            'ratio': result,
            'dict_feedback_pre': dict_feedback_pre,
            'dict_feedback_test': dict_feedback_test
        }
        return train_set


modelEvaluation = modelEvaluation()
modelEvaluation.evalue()
print('Done!')
