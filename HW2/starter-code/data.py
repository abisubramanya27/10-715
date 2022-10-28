import numpy as np
from sklearn.datasets import make_blobs
from soft_svm import SoftSVM
import matplotlib.pyplot as plt

# Generates data for the problem.
def get_data():
    std = 3.1
    train_data = make_blobs(n_samples=10_000, n_features=2, centers=2, cluster_std=std, random_state=1)
    test_data = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=std, random_state=1)
    return prepare_data(train_data), prepare_data(test_data)

# When applied to train_data or test_data, prepares the weight vector and transforms labels
def prepare_data(data):
    # Add a constant 1 feature to X and change y labels to {-1,1}
    X = np.hstack([np.ones((len(data[0]),1)), data[0]])
    y = 1*(data[1]==1) -1*(data[1]==0)
    return X, y

def plot_samples(X_train, y_train, svm, random_seed = 1, frac = 0.1):
    _ = plt.figure(figsize=(5,6))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    n = X_train.shape[0]
    np.random.seed(random_seed)
    idx = np.random.choice(n, (int(frac*n),), replace=False)
    is_sup_vector = ((X_train @ svm.w) * y_train) <= 1
    for i in idx:
        marker = 'o' if y_train[i] == -1 else 'x'
        if is_sup_vector[i]:
            continue
        label = 'Label -1 | Not a Support Vector' if y_train[i] == -1 else 'Label 1 | Not a Support Vector'
        color = '#1f77b4'
        plt.scatter(X_train[i][1], X_train[i][2], marker=marker, c=color, label=label)
    for i in idx:
        marker = 'o' if y_train[i] == -1 else 'x'
        if not is_sup_vector[i]:
            continue
        label = 'Label -1 | Support Vector' if y_train[i] == -1 else 'Label 1 | Support Vector'
        color = 'r'
        plt.scatter(X_train[i][1], X_train[i][2], marker=marker, c=color, label=label)
        
    x1_min, x1_max = np.min(X_train[:,1]), np.max(X_train[:,1])
    x = np.linspace(x1_min, x1_max, 100)
    plt.plot(x, (-x*svm.w[1] - svm.w[0])/svm.w[2], c='m', label='Decision Boundary')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
(X_train, y_train), (X_test, y_test) = get_data()
svm = SoftSVM(50)
svm.train(X_train, y_train, 10000, 1e-5)
final_train_loss, final_train_acc = svm.loss(X_train, y_train), svm.accuracy(X_train, y_train)
print(final_train_loss, final_train_acc)
final_test_loss, final_test_acc = svm.loss(X_test, y_test), svm.accuracy(X_test, y_test)
print(final_test_loss, final_test_acc)
# plot_samples(X_train, y_train, svm)