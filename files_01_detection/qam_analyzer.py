import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special as sp
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

'''
Q function expressed in terms of the error function (https://en.wikipedia.org/wiki/Q-function).
'''
def _qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

"""
Theoretical symbol error probability for squared M-QAM.
"""
def theoretical_ser(M, SNR_db):
    SNR_l = 10**(SNR_db/10) #from dB to linear scale
    Pe = 4*(1-(1/np.sqrt(M)))*_qfunc(np.sqrt(3*SNR_l/(M-1))) \
         - 4*(1-(1/np.sqrt(M)))**2 * _qfunc(np.sqrt(3*SNR_l/(M-1)))**2
    return Pe

def ser(clf, X, y):
    """ Calculate the misclassification rate, which
        coincides with the symbol error rate (SER) for QAM transmission.
    """
    y_pred = clf.predict(X)
    ser    = np.sum(y != y_pred)/len(y)

    return ser

def plot_confusion_matrix(clf, X, y, num_classes):
    """ Plot the confusion matrix
    """
    y_pred   = clf.predict(X)
    conf_mtx = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10,6))
    sns.heatmap(conf_mtx, cmap=sns.cm.rocket_r, square=True, linewidths=0.1,
                annot=True, fmt='d', annot_kws={"fontsize": 8})
    plt.tick_params(axis='both', which='major', labelsize=10,
                    bottom=False, top=False, left=False,
                    labelbottom=False, labeltop=True)
    plt.yticks(rotation=0)


    plt.show()


def plot_decision_boundary(classifier, X, y, legend=False, plot_training=True):
    """ Plot the classifier decision regions
    """
    num_classes = int(np.max(y))+1 #e.g. 16 for QAM-16
    axes = [np.min(X[:,0]), np.max(X[:,0]),np.min(X[:,1]), np.max(X[:,1])]
    #print(axes)
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = classifier.predict(X_new).reshape(x1.shape)

    # Set different color for each class
    custom_cmap = cm.get_cmap('tab20')
    colors = custom_cmap.colors[:num_classes]
    levels = np.arange(num_classes + 2) - 0.5

    plt.contourf(x1, x2, y_pred, levels=levels, colors=colors, alpha=0.3)

    if plot_training:
        for ii in range(num_classes):
            selected_indices = np.argwhere(y==ii)
            selected_indices = selected_indices.reshape((-1,))
            plt.plot(X[selected_indices, 0], X[selected_indices, 1], "o",
                     c=colors[ii], label=f'{ii}')
        #plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        #plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        #plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        #plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(title='Classes', bbox_to_anchor=(1, 1), loc='upper left',
                   ncol=2, handleheight=2, labelspacing=0.05, frameon=False)
    plt.show()


def main():
    # file_name = 'qam_crazy.csv'
    file_name = 'qam_awgn.csv'
    with open(file_name, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    data = np.array(data, dtype=float)
    #print(data.shape)

    X = data[:,:-1] # petal length and width
    y = data[:,-1]  # Labels

    classifier = DecisionTreeClassifier(max_depth=20, random_state=42)
    classifier.fit(X, y)

    classifier_ser = ser(classifier, X, y)
    print(f'SER: {classifier_ser*100:.3f} %')

    plot_decision_boundary(classifier, X, y, legend=False, plot_training=True)

if __name__ == '__main__':
    main()

