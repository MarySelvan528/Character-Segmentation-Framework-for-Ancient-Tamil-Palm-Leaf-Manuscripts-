import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = np.arange(len(Terms))
    Algorithm = ['TERMS', 'CO', 'WaOA', 'SOA', 'FA', 'PROPOSED']
    Classifier = ['TERMS', 'ANN', 'SVM', 'BL', 'ENSEMBLE', 'PROPOSED']
    value = eval[4, :, 4:]

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('---------------------------------------- Algorithm Comparison',
          '----------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    print('---------------------------------------- Classifier Comparison',
          '----------------------------------------')
    print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if j == 9:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4] * 100

        fig = plt.figure()
        X = np.arange(Graph.shape[0])
        ax = fig.add_axes([0.15, 0.25, 0.75, 0.6])
        ax.plot(X, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue',
                markersize=12, label="CO-EMLS")
        ax.plot(X, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red',
                markersize=12, label="WaOA-EMLS")
        ax.plot(X, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green',
                markersize=12, label="SOA-EMLS")
        ax.plot(X, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow',
                markersize=12, label="FA-EMLS")
        ax.plot(X, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan',
                markersize=12, label="HSFA-EMLS")
        plt.xticks(X, ('LBP', 'LTRP', 'LBP + LTRP', 'AE Deep feature',
                       'LBP + LTRP + \nAE Deep feature'), rotation=45)  # , rotation=45
        # plt.xlabel('Features')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='best')
        path1 = "./Results/%s_alg.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.25, 0.75, 0.6])
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="ANN")
        ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="SVM")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="BL")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="EMLS")
        ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="HSFA-EMLS")
        plt.xticks(X, ('LBP', 'LTRP', 'LBP + LTRP', 'AE Deep feature',
                       'LBP + LTRP + \nAE Deep feature'), rotation=45)  # , rotation=45
        # plt.xlabel('Features')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='best')
        path1 = "./Results/%s_mtd_1.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CO', 'WaOA', 'SOA', 'FA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(25)
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='CO-EMLS')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label='WaOA-EMLS')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='SOA-EMLS')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='FA-EMLS')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='HSFA-EMLS')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


if __name__ == '__main__':
    plot_results()
    plotConvResults()
