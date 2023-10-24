import matplotlib.pyplot as plt
import numpy as np


class tracer:
    def __init__(self, filenames, epochs, bias, curve_names):
        self.data_files = filenames
        self.data_epochs = epochs
        self.data_labels = curve_names
        self.data_bias = bias
        self.train_data = []
        self.valid_data = []
        self.train_loss = []
        self.valid_loss = []

        self.train_data_bcmk = []
        self.valid_data_bcmk = []
        self.train_loss_bcmk = []
        self.valid_loss_bcmk = []
        self.data_epochs_bcmk = []
        self.data_labels_bcmk = []
        self.data_bias_bcmk = []

    def read_data(self):
        count = 0
        for name in self.data_files:
            self.train_data.append([])
            self.valid_data.append([])
            self.train_loss.append([])
            self.valid_loss.append([])

            with open(name, "r") as file:

                for line in file:

                    line = line.split(";")

                    if self.data_bias[count] == 6:
                        self.train_data[-1].append(float(line[1]))
                        self.valid_data[-1].append(float(line[4]))
                        self.train_loss[-1].append(float(line[2]))
                        self.valid_loss[-1].append(float(line[5]))
                    else:
                        self.train_data[-1].append(float(line[1]))
                        self.valid_data[-1].append(float(line[3]))
                        self.train_loss[-1].append(float(line[2]))
                        self.valid_loss[-1].append(float(line[4]))
            count += 1
            file.close()

    def explore_best_accuracy_result(self):
        max_val = []
        mean_val = []
        idx = []
        for i in range(len(self.data_files)):
            comb = len(self.valid_data[i]) // int(self.data_epochs[i])
            # print(len(valid_data))
            buf_max = []
            buf_mean = []
            for j in range(comb):
                buf_max.append(max(self.valid_data[i][j * self.data_epochs[i]: (j + 1) * self.data_epochs[i]]))
                buf_mean.append(np.mean(np.array(self.valid_data[i][j * self.data_epochs[i]: (j + 1) * self.data_epochs[i]])))

            max_val.append(max(buf_max))
            mean_val.append(max(buf_mean))
            idx.append((buf_max.index(max_val[-1]), buf_mean.index(mean_val[-1])))

        for i in range(len(self.data_files)):
            print(self.data_files[i], ": max vall_acc - ", max_val[i], ", mean vall_acc - ", mean_val[i], ", indexes - ",
                  idx[i])

    def explore_best_loss_result(self):
        min_val = []
        mean_val = []
        idx = []
        for i in range(len(self.data_files)):
            comb = len(self.valid_loss[i]) // int(self.data_epochs[i])
            # print(len(valid_data))
            buf_min = []
            buf_mean = []
            for j in range(comb):
                buf_min.append(min(self.valid_loss[i][j * self.data_epochs[i]: (j + 1) * self.data_epochs[i]]))
                buf_mean.append(np.mean(np.array(self.valid_loss[i][j * self.data_epochs[i]: (j + 1) * self.data_epochs[i]])))

            min_val.append(min(buf_min))
            mean_val.append(min(buf_mean))
            idx.append((buf_min.index(min_val[-1]), buf_mean.index(mean_val[-1])))

        for i in range(len(self.data_files)):
            print(self.data_files[i], ": min vall_loss - ", min_val[i], ", mean vall_loss - ", mean_val[i], ", indexes - ",
                  idx[i])

    def set_benchmark_curves(self, idx):
        for i in idx:
            self.train_data_bcmk.append(self.train_data[i])
            self.valid_data_bcmk.append(self.valid_data[i])
            self.train_loss_bcmk.append(self.train_loss[i])
            self.valid_loss_bcmk.append(self.valid_loss[i])
            self.data_epochs_bcmk.append(self.data_epochs[i])
            self.data_labels_bcmk.append(self.data_labels[i])
            self.data_bias_bcmk.append(self.data_bias[i])

        for index in sorted(idx, reverse=True):
            del self.data_epochs[index]
            del self.data_labels[index]
            del self.data_bias[index]
            del self.train_data[index]
            del self.valid_data[index]
            del self.train_loss[index]
            del self.valid_loss[index]




epochs =[ 100, 70, 70]
num_char = [6,4,4]
filenames = ["C:/Users/ADostovalova/Desktop/work/функция_активации/24_10_23/ model_classify_cifar10_mixture_3.csv",
             "C:/Users/ADostovalova/Desktop/work/функция_активации/24_10_23/ model_classify_cifar_resnet56_mixture_1.csv",
             "C:/Users/ADostovalova/Desktop/work/функция_активации/24_10_23/ model_classify_cifar__resnet20_mixture_0.csv"]
curv_names = ['efficientNet_mix', 'resnet56_mix', 'resnet20_mix']
tr = tracer(filenames,epochs, num_char, curv_names)
tr.read_data()
tr.explore_best_loss_result()
tr.explore_best_accuracy_result()

