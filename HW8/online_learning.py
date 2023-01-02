import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OnlineLearning:
    def __init__(self, eps, d, seed=0):
        self._seed = seed
        self.eps = eps
        self._d = d
        self.w = np.ones((d,))

    def fit(self, x, y_true):
        pred = np.random.choice(x, p=self.w/np.sum(self.w) if np.sum(self.w) > 0 else None)
        self.w -= self.eps * np.not_equal(x, y_true) * self.w
        return pred

    def fit_baseline(self, X_prev, Y_prev, x_cur):
        if len(X_prev) == 0:
            loss = np.zeros((self._d,))
        else:
            loss = np.sum(X_prev != Y_prev.reshape(len(Y_prev), 1), axis=0)
        return x_cur[np.argmin(loss)]

    def fit_horizon(self, X, Y_true, baseline=False):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            Y_true = Y_true.to_numpy()
        T = len(X)
        self.predictions = []
        self.baseline_preds = []
        np.random.seed(self._seed)
        for i in range(T):
            self.w /= np.sum(self.w) # Normalizing (comment if not needed)
            self.predictions.append(self.fit(X[i], Y_true[i]))
            if baseline:
                self.baseline_preds.append(self.fit_baseline(X[:i], Y_true[:i], X[i]))

        self.predictions = np.array(self.predictions)
        self.baseline_preds = np.array(self.baseline_preds)
        self.calc_regret(X, Y_true, baseline)
        return self.predictions

    def calc_regret(self, X, Y_true, baseline):
        self.regret = np.mean(np.not_equal(self.predictions, Y_true))
        best_expert_regret = np.min(np.mean(X != Y_true.reshape((len(Y_true), 1)), axis=0))
        self.regret -= best_expert_regret
        if baseline:
            self.baseline_regret = np.mean(np.not_equal(self.baseline_preds, Y_true))
            self.baseline_regret -= best_expert_regret

def plot(eps_set, mean, std, best_expert_regret):
    plt.figure(0)
    plt.plot(eps_set, mean, c='r', label='Mean Avg Regret')
    plt.plot(eps_set, np.maximum(mean-std, 0), linestyle='-.', label='Mean - Std')
    plt.plot(eps_set, mean+std, linestyle='-.', label='Mean + Std')
    plt.fill_between(eps_set, np.maximum(mean-std, 0), mean+std, color='r', alpha=.1)
    plt.axhline(y=best_expert_regret, color='b', linestyle='-', label='Baseline Avg Regret')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('Average Regret')
    plt.legend(loc='best')
    plt.savefig('regret1.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    df = pd.read_csv('online_data.csv', sep=', ', index_col='T')
    X_train = df.loc[:, df.columns != 'Label']
    Y_train = df['Label']
    regret_mean, regret_std, baseline_regret = [], [], None
    eps_set = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    for eps in eps_set:
        regrets = []
        b_regrets = []
        for seed in [0, 100, 200, 300, 400, 500]:
            algo = OnlineLearning(eps, 10, seed)
            algo.fit_horizon(X_train, Y_train, True if baseline_regret is None else False)
            regrets.append(algo.regret)
            if baseline_regret is None:
                b_regrets.append(algo.baseline_regret)

        regret_mean.append(np.mean(regrets))
        regret_std.append(np.std(regrets))
        if baseline_regret is None:
            baseline_regret = np.mean(b_regrets)
        print(f'For eps: {eps: >5} | mean-regret: {regret_mean[-1]: >8.4f} | '
              f'std-regret: {regret_std[-1]: >8.4f} | baseline regret: {baseline_regret: >8.4f}')

    plot(eps_set, np.array(regret_mean), np.array(regret_std), baseline_regret)
    X = X_train.to_numpy()
    Y_true = Y_train.to_numpy()
    best_expert_regret = np.min(np.mean(X != Y_true.reshape((len(Y_true), 1)), axis=0))
    print(f'Best Expert Average Regret (at the end): {best_expert_regret}')