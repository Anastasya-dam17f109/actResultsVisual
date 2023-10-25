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
                        print(line)
                        self.train_data[-1].append(float(line[1]))
                        self.valid_data[-1].append(float(line[4]))
                        self.train_loss[-1].append(float(line[2]))
                        self.valid_loss[-1].append(float(line[5]))
                    else:
                        #print(line)
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
        print("two first", (max_val[0] - max_val[1]) * 100)

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
        print("two first", (min_val[0]-min_val[1])*100)

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

    def print_the_comparison(self, idx_mix, idx_bench):
        if self.data_epochs[idx_mix] == self.data_epochs_bcmk[idx_bench]:
            train_data = [[],[]]
            valid_data = [[],[]]
            train_loss = [[],[]]
            valid_loss = [[],[]]
            N = len(self.train_data_bcmk[idx_bench]) // self.data_epochs_bcmk[idx_bench]
            for i in range(self.data_epochs_bcmk[idx_bench]):
                buf0, buf1, buf2, buf3 = 0,0,0,0
                for j in range(N):
                    buf0 += self.train_data_bcmk[idx_bench][j*self.data_epochs_bcmk[idx_bench] + i]
                    buf1 += self.train_loss_bcmk[idx_bench][j * self.data_epochs_bcmk[idx_bench] + i]
                    buf2 += self.valid_data_bcmk[idx_bench][j * self.data_epochs_bcmk[idx_bench] + i]
                    buf3 += self.valid_loss_bcmk[idx_bench][j * self.data_epochs_bcmk[idx_bench] + i]

                train_data[0].append(buf0 / N)
                valid_data[0].append(buf2 / N)
                train_loss[0].append(buf1 / N)
                valid_loss[0].append(buf3 / N)
            buf4 = 0
            med_gelu = 0
            for j in range(N):
                buf4 += max(self.valid_data_bcmk[idx_bench][j * self.data_epochs_bcmk[idx_bench] : (j+1) * self.data_epochs_bcmk[idx_bench]])
                gl = self.valid_data_bcmk[idx_bench][j * self.data_epochs_bcmk[idx_bench] : (j+1) * self.data_epochs_bcmk[idx_bench]].copy()
                gl.sort()
                med_gelu += gl[self.data_epochs[idx_bench] // 2]
            buf4 /=N

            med_gelu /=N
            N = len(self.train_data[idx_mix]) // self.data_epochs[idx_mix]
            for i in range(self.data_epochs[idx_mix]):
                buf0, buf1, buf2, buf3 = 0, 0, 0, 0
                for j in range(N):
                    buf0 += self.train_data[idx_mix][j * self.data_epochs[idx_mix] + i]
                    buf1 += self.train_loss[idx_mix][j * self.data_epochs[idx_mix] + i]
                    buf2 += self.valid_data[idx_mix][j * self.data_epochs[idx_mix] + i]
                    buf3 += self.valid_loss[idx_mix][j * self.data_epochs[idx_mix] + i]
                train_data[1].append(buf0 / N)
                valid_data[1].append(buf2 / N)
                train_loss[1].append(buf1 / N)
                valid_loss[1].append(buf3 / N)

            buf5 = 0
            med_mix = 0
            for j in range(N):
                buf5 += max(self.valid_data[idx_mix][
                            j * self.data_epochs[idx_mix]: (j + 1) * self.data_epochs[idx_mix]])
                mx = self.valid_data[idx_mix][
                            j * self.data_epochs[idx_mix]: (j + 1) * self.data_epochs[idx_mix]].copy()
                mx.sort()
                med_mix += mx[self.data_epochs[idx_mix] // 2]
            buf5 /= N

            med_mix /= N
            print("gelu: ", buf4*100, " mixture: ", buf5*100, " delta:  " , (buf5-buf4)*100)
            print("gelu: ", med_gelu*100, " mixture: ", med_mix*100, " delta:  ", (med_mix-med_gelu)*100)
            x = np.linspace(1, self.data_epochs[idx_mix], self.data_epochs[idx_mix])

            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 2)  # задаем сетку 3х4
            ax = [None for _ in range(4)]

            ax[0] = fig.add_subplot(gs[0, 0])
            ax[0].plot(x, train_data[0], color="blue", label=self.data_labels_bcmk[idx_bench])
            ax[0].plot(x, train_data[1], color="red", label= self.data_labels[idx_mix])  # дл#для каждого графика задаем позицию и размер
            ax[0].set_title('train acc')
            ax[0].legend()  # через элементы сетки
            ax[1] = fig.add_subplot(gs[1, 0])
            ax[1].plot(x, valid_data[0], color="blue", label=self.data_labels_bcmk[idx_bench])
            ax[1].plot(x, valid_data[1], color="red", label=self.data_labels[idx_mix])
            ax[1].set_title('valid acc')
            ax[1].legend()
            ax[2] = fig.add_subplot(gs[1, 1])
            ax[2].plot(x, train_loss[0], color="blue", label=self.data_labels_bcmk[idx_bench])
            ax[2].plot(x, train_loss[1], color="red", label=self.data_labels[idx_mix])
            ax[2].set_title('train loss')
            ax[2].legend()
            ax[3] = fig.add_subplot(gs[0, 1])
            ax[3].plot(x, valid_loss[0], color="blue", label=self.data_labels_bcmk[idx_bench])
            ax[3].plot(x, valid_loss[1], color="red", label=self.data_labels[idx_mix])
            ax[3].set_title('valid loss')
            ax[3].legend()
        else:
            print("data cannot be compared")




#epochs =[ 100, 70, 70]
#num_char = [6,4,4]
#filenames = ["C:/Users/ADostovalova/Desktop/work/функция_активации/24_10_23/ model_classify_cifar10_mixture_3.csv",
            # "C:/Users/ADostovalova/Desktop/work/функция_активации/24_10_23/ model_classify_cifar_resnet56_mixture_1.csv",
            # "C:/Users/ADostovalova/Desktop/work/функция_активации/24_10_23/ model_classify_cifar__resnet20_mixture_0.csv"]
#curv_names = ['efficientNet_mix', 'resnet56_mix', 'resnet20_mix']


epochs =[ 70, 70]
num_char = [4,4]
filenames = [#"E:/ФА_статьи/best/archive/model_classify_cifar_resnet56_mixture_2.csv",
            #"E:/ФА_статьи/best/archive/model_classify_cifar_resnet56_gelu.csv",
"E:/ФА_статьи/best/archive/model_classify_cifar__resnet20_mixture_1.csv",
"E:/ФА_статьи/best/archive/model_classify_cifar__resnet20_gelu.csv"
             ]
curv_names = ['resnet56_mix', 'resnet56_gelu']
tr = tracer(filenames,epochs, num_char, curv_names)
tr.read_data()
#tr.explore_best_loss_result()
tr.explore_best_accuracy_result()
tr.set_benchmark_curves([1])
tr.print_the_comparison(0, 0)
plt.show()