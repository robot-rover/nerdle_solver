ESTIMATOR_DATA_PATH = 'estimator.csv'
ESTIMATOR_PATH = 'estimator_params.npy'
NUM_BINS = 20

if __name__ == "__main__":
    import numpy as np
    from numpy.polynomial.polynomial import polyfit
    import matplotlib.pyplot as plt

    data = np.genfromtxt(ESTIMATOR_DATA_PATH, delimiter=',', usecols=(1,0))
    # data[sample] = (entropy, num_guess)
    data_unique = np.unique(data, axis=0)

    # remove 1 entries
    # data = data[data[:,1] > 1]

    # print(data.shape)
    # print(data[:10])

    counts, bins = np.histogram(data[:,0], bins=NUM_BINS)
    counts[counts == 0] += 1 # Prevent NaN on zero count divisions
    weights, _ = np.histogram(data[:,0], weights=data[:,1], bins=bins)

    estimator = weights / counts

    reg_pnts = np.stack((bins[:-1], estimator), axis=-1)
    reg_pnts = reg_pnts[reg_pnts[:,1] != 0]
    regression = np.zeros((3,))
    regression[:2] = polyfit(np.log(reg_pnts[:,0]), reg_pnts[:,1], 1)
    regression[2] = np.exp((1-regression[0])/regression[1])
    np.save(ESTIMATOR_PATH, regression, allow_pickle=False)
    print("Regression:", regression)
    print(f'y = {regression[1]} * log(x) + {regression[0]}')

    x_draw = np.linspace(bins[0], bins[-1], 100)
    y_draw = regression[1] * np.log(x_draw) + regression[0]

    plt.title("Remaining Uncertainty vs Guesses Required")
    plt.ylabel("# Guesses left to Win")
    plt.xlabel("Uncertainty (bits)")
    plt.hist(bins[:-1], bins, weights=estimator, edgecolor='k', label='Binned')
    plt.scatter(data_unique[:,0], data_unique[:,1], c='orange', marker='.', label='Raw')
    plt.plot(x_draw, y_draw, c='r', label='Fit')
    plt.legend()
    plt.show()
