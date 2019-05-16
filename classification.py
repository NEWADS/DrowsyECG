import scipy.io as sio
import numpy as np
import RelativeMethods as rm
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def main(c):
    # loading the dataset...
    tf1, tf0 = rm.data_generator("./dataset/Transferred/icp")
    hrv1, hrv0 = rm.data_generator("./dataset/HRV")

    # applying isomap to transfer learning features...
    tf1, tf0 = rm.isomap(tf1, tf0)

    # putting them together...
    feature1 = np.concatenate((hrv1, tf1), axis=1)
    # feature1 = hrv1
    # feature1 = tf1
    y1 = np.ones(feature1.shape[0])
    feature0 = np.concatenate((hrv0, tf0), axis=1)
    # feature0 = hrv0
    # feature0 = tf0
    y0 = np.zeros(feature0.shape[0])
    x = np.concatenate((feature1, feature0))
    y = np.concatenate((y1, y0))

    # shuffling dataset...
    np.random.seed(114)
    x, y = rm.shuffle(x, y)
    cv = StratifiedKFold(n_splits=15)

    # creating classifier...
    # employing random forest
    # rfc = RFC(n_estimators=114)  # for test
    rfc = RFC(n_estimators=1)  # For 16 demensions
    # svm
    svm = SVC(C=c, cache_size=200, class_weight=None,
              decision_function_shape='ovr', gamma=1.25e-05, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    # Deploying SGDlearner
    # svm = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
    #        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
    #        learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
    #        n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
    #        shuffle=True, tol=None, verbose=0, warm_start=False)
    # 10-Fold Validation
    acc_rfc = []
    f1_rfc = []
    auc_rfc = []
    acc_svm = []
    f1_svm = []
    auc_svm = []
    for train, test in cv.split(x, y):
        # Predict
        y_svm_pred = svm.fit(x[train], y[train]).predict(x[test])
        y_rfc_pred = rfc.fit(x[train], y[train]).predict(x[test])
        # Scoring
        acc_rfc.append(metrics.accuracy_score(y[test], y_rfc_pred))
        f1_rfc.append(metrics.f1_score(y[test], y_rfc_pred))
        auc_rfc.append(metrics.roc_auc_score(y[test], y_rfc_pred))
        acc_svm.append(metrics.accuracy_score(y[test], y_svm_pred))
        f1_svm.append(metrics.f1_score(y[test], y_svm_pred))
        auc_svm.append(metrics.roc_auc_score(y[test], y_svm_pred))

    print('16-d dimension reduced icp features classification results:')
    rslttb = PrettyTable([" ", "Accuracy", "F1", "AUC"])
    rslttb.add_row(["SVM",
                    np.array(acc_svm).mean(),
                    np.array(f1_svm).mean(),
                    np.array(auc_svm).mean()])
    rslttb.add_row(['RF',
                    np.array(acc_rfc).mean(),
                    np.array(f1_rfc).mean(),
                    np.array(auc_rfc).mean()])
    print(rslttb)


if __name__ == "__main__":
    main(c=5000)  # 16-d
    # for i in range(1000, 5000, 200):
    #     main(c=i)
    # main(c=25)  # 8-d
