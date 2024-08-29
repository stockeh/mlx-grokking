import numpy as np
import mlx.core as mx


def grokking_data(p: int, op: str = '/', train_fraction: float = 0.5):
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p-2, p)) % p,
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p
    }

    if op not in operations:
        raise ValueError(
            "Unsupported operation, choose from ['*', '/', '+', '-']")

    X = np.array([(a, b) for a in range(p)
                 for b in range(1 if op == '/' else 0, p)])
    T = np.array([operations[op](a, b) for a, b in X])

    embed = {'*': p, '/': p, '+': p, '-': p, '=': p + 1}
    X = np.array([
        [a, embed[op], b, embed['=']]
        for (a, b) in X
    ])

    n_train = int(train_fraction * len(X))
    inds = np.random.permutation(len(X))
    Xtrain, Ttrain = X[inds[:n_train]], T[inds[:n_train]]
    Xtest, Ttest = X[inds[n_train:]], T[inds[n_train:]]

    return mx.array(Xtrain), mx.array(Ttrain), mx.array(Xtest), mx.array(Ttest)


if __name__ == '__main__':
    Xtrain, Ttrain, Xtest, Ttest = grokking_data(
        11, op='/', train_fraction=0.5)
    print(Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape)
    print(Xtrain[0], Ttrain[0])
