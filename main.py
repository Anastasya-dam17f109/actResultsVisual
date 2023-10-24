# This is a sample Python script.
import matplotlib.pyplot as plt
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

epochs = 100
filenames = ["C:/Users/ADostovalova/Desktop/work/функция_активации/ model_classify_cifar_resnet_gelu1.csv",
             "C:/Users/ADostovalova/Desktop/work/функция_активации/ model_classify_cifar_resnet_mix1.csv"]
train_data = []
valid_data =[]
train_loss = []
valid_loss = []
for name in filenames:
    train_data.append([])
    valid_data.append([])
    train_loss.append([])
    valid_loss.append([])
    with open(name, "r") as file:

        for line in file:
            line = line.split(";")
            if(line[1] != "accuracy"):

                train_data[-1].append(float(line[1]))
                valid_data[-1].append(float(line[3]))
                train_loss[-1].append(float(line[2]))
                valid_loss[-1].append(float(line[4]))
        #train_data[-1] = train_data[-1][1:]
        #valid_data[-1]=valid_data[-1][1:]
        #train_loss[-1] = train_loss[-1][1:]
        #valid_loss[-1] = valid_loss[-1][1:]
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
x = np.linspace(1,100,100)
print( train_data[1])
fig = plt.figure(figsize=(16, 8))

gs = fig.add_gridspec(2, 2)           #задаем сетку 3х4

ax = [None for _ in range(4)]

ax[0] = fig.add_subplot(gs[0, 0])
ax[0].plot(x,train_data[0], color= "blue",label="resnet_gelu")
ax[0].plot(x,train_data[1], color= "red", label="resnet_lstm_pure")#дл#для каждого графика задаем позицию и размер
ax[0].set_title('train acc')
ax[0].legend()#через элементы сетки
ax[1] = fig.add_subplot(gs[1, 0])
ax[1].plot(x,valid_data[0], color= "blue",label="resnet_gelu")
ax[1].plot(x,valid_data[1], color= "red", label="resnet_lstm_pure")
ax[1].set_title('valid acc')
ax[1].legend()
ax[2] = fig.add_subplot(gs[1, 1])
ax[2].plot(x,train_loss[0], color= "blue",label="resnet_gelu")
ax[2].plot(x,train_loss[1], color= "red", label="resnet_lstm_pure")
ax[2].set_title('train loss')
ax[2].legend()
ax[3] = fig.add_subplot(gs[0, 1])
ax[3].plot(x,valid_loss[0], color= "blue",label="resnet_gelu")
ax[3].plot(x,valid_loss[1], color= "red", label="resnet_lstm_pure")
ax[3].set_title('valid loss')
ax[3].legend()
plt.show()