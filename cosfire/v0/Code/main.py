import parametros as params
import numpy as np
import COSFIRE
import pickle
import math
import gzip
import cv2
import mnist_reader
from sklearn import svm
import matplotlib.pyplot as plt


def main():
    # Example with a simple pattern
    ejemplo1()
    # Example with a simple pattern invariant to reflection
    # ejemplo2()


def ejemplo1():
    im1 = cv2.imread("data/patron1.jpg", 0)
    a = COSFIRE.COSFIRE(im1)
    params.parametros1(a)
    a.fit()
    im2 = cv2.imread("data/prueba1.jpg", 0)
    r = a.transform(im2)
    viewPattern(r, im2, 20)
    cv2.imshow("Response", r)


# %%
def ejemplo2():
    im1 = cv2.imread("data/patron1.jpg", 0)
    a = COSFIRE.COSFIRE(im1)
    params.parametros2(a)
    a.fit()
    im2 = cv2.imread("data/prueba1.jpg", 0)
    r = a.transform(im2)
    viewPattern(r, im2, 25)
    cv2.imshow("Response", r)


def SenalTraf(namePattern, nameImage):
    im1 = cv2.imread(namePattern, 0)
    im2 = cv2.imread(nameImage, 0)
    a = COSFIRE.COSFIRE(im1)
    params.parametros3(a)
    a.fit()
    r = a.transform(im2)
    viewPattern(r, im2, 15)
    cv2.imshow("Response", r)
    return r, im2


def ClassDIGITOS():
    f = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    N = 2  # Numero de digitos a considerar
    M = 25  # Numero de filtros COSFIRE por digito

    list1 = []  # Lista de imagenes
    for i in range(N):
        l = []
        cont = 0
        j = 0
        while 1 and cont != M:
            if train_set[1][j] == i:
                im = train_set[0][j].reshape(28, -1).copy()
                l.append(im)
                cont += 1
            j += 1
        list1.append(l)
    cont = 0
    descriptores = []
    for i in range(N):
        for j in range(M):
            print(cont)
            cont += 1
            a = COSFIRE.COSFIRE(list1[i][j])
            params.parametros6(a)
            a.coorX = np.random.randint(4, 24)
            a.coorY = np.random.randint(4, 24)
            a.fit()
            descriptores.append(a)
    aux = np.zeros(10)
    cont = 0
    numEntr = 50
    H = np.zeros((numEntr, M * N))
    y = np.zeros((numEntr))
    for i in range(numEntr):
        print(i)
        while 1:
            if cont >= 5000:
                cont = 0
            if train_set[1][cont] >= N or aux[train_set[1][cont]] >= numEntr / N + 1:
                cont += 1
            else:
                aux[train_set[1][cont]] += 1
                break
        for j in range(M * N):
            im = train_set[0][cont].reshape(28, -1)
            if len(descriptores[j].operator) != 0:
                H[i][j] = descriptores[j].transform(im).max()
            y[i] = train_set[1][cont]
        cont += 1

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(H, y)
    numPrueba = 25
    H1 = np.zeros((numPrueba, M * N))
    y1 = np.zeros((numPrueba))
    cont = 0
    aux = np.zeros(10)
    for i in range(numPrueba):
        print(i)
        while 1:
            if cont >= 5000:
                cont = 0
            if test_set[1][cont] >= N or aux[test_set[1][cont]] >= numEntr / N:
                cont += 1
            else:
                aux[test_set[1][cont]] += 1
                break
        for j in range(M * N):
            im = test_set[0][cont].reshape(28, -1)
            if len(descriptores[j].operator) != 0:
                H1[i][j] = descriptores[j].transform(im).max()
            y1[i] = test_set[1][cont]
        cont += 1
    return clf, H, H1, y, y1


def ClassFASHION():
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    N = 10  # Numero de digitos a considerar
    M = 200  # Numero de filtros COSFIRE por digito

    list1 = []  # Lista de imagenes
    for i in range(N):
        l = []
        cont = 0
        j = 0
        while 1 and cont != M:
            if y_train[j] == i:
                im = X_train[j].reshape(28, 28).copy()
                l.append(im)
                cont += 1
            j += 1
        list1.append(l)
    cont = 0
    descriptores = []
    for i in range(N):
        for j in range(M):
            print(cont)
            cont += 1
            a = COSFIRE.COSFIRE(list1[i][j])
            params.parametros6(a)
            a.coorX = np.random.randint(4, 24)
            a.coorY = np.random.randint(4, 24)
            a.fit()
            descriptores.append(a)
    aux = np.zeros(10)
    cont = 0
    numEntr = 300
    H = np.zeros((numEntr, M * N))
    y = np.zeros((numEntr))
    for i in range(numEntr):
        print(i)
        while 1:
            if cont >= 5000:
                cont = 0
            if y_train[cont] >= N or aux[y_train[cont]] >= numEntr / N + 1:
                cont += 1
            else:
                aux[y_train[cont]] += 1
                break
        for j in range(M * N):
            im = X_train[cont].reshape(28, 28)
            if len(descriptores[j].operator) != 0:
                # print "-> ", descriptores[j].applyCOSFIRE(im).max()
                dm = descriptores[j].transform(im).max()
                if math.isnan(dm):
                    dm = 0
                H[i][j] = dm
            y[i] = y_train[cont]
        cont += 1

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    print(np.where(np.isnan(H)))
    print(np.where(np.isnan(y)))
    clf.fit(H, y)
    numPrueba = 100
    H1 = np.zeros((numPrueba, M * N))
    y1 = np.zeros((numPrueba))
    cont = 0
    aux = np.zeros(10)
    for i in range(numPrueba):
        print(i)
        while 1:
            if cont >= 5000:
                cont = 0
            if y_test[cont] >= N or aux[y_test[cont]] >= numEntr / N:
                cont += 1
            else:
                aux[y_test[cont]] += 1
                break
        for j in range(M * N):
            im = X_test[cont].reshape(28, 28)
            if len(descriptores[j].operator) != 0:
                dm = descriptores[j].transform(im).max()
                if math.isnan(dm):
                    dm = 0
                H1[i][j] = descriptores[j].transform(im).max()
            y1[i] = y_test[cont]
        cont += 1
    return clf, H, H1, y, y1


def viewPattern(im1, im2, radio):
    mask = np.zeros(im1.shape)
    cont = 1
    l = []
    for i in range(im1.shape[0] - 2):
        for j in range(im2.shape[1] - 2):
            if im1[i + 1][j + 1] != 0:
                if (
                    im1[i + 2][j + 1] == 0
                    and im1[i][j + 1] == 0
                    and im1[i + 1][j + 2] == 0
                    and im1[i + 1][j] == 0
                ):
                    cont = cont + 1
                    l.append((i + 1, j + 1))
                mask[i + 1][j + 1] = cont
    phi = np.arange(360) * np.pi / 180.0
    for item in l:
        for j in range(len(phi)):
            im2[int(item[0] - radio * np.sin(phi[j]))][
                int(item[1] + radio * np.cos(phi[j]))
            ] = 0
    cv2.imshow("dasd", im2)


if __name__ == '__main__':
    main()
    print('This is the end, my friend')