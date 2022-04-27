from matplotlib import pyplot as plt
import numpy as np

NUM_BINS=10

data = np.genfromtxt('entropy.csv', delimiter=',')
# data[sample] = (guess_num in <1..>, uncertainty, information)

# datanz = data[data[:,1] != 0]

moves = np.unique(data[:,0])
f, axs = plt.subplots(2, len(moves), sharex='col')
f.set_size_inches(13, 7.5)

for idx, move in enumerate(moves):
    move_data = data[data[:,0] == move, :]
    ihist, ibins = np.histogram(move_data[:,2], bins=NUM_BINS, density=True)
    axs[1,idx].hist(ibins[:-1], ibins, weights=ihist*(ibins[1]-ibins[0]))
    axs[1,idx].set_title(f'Guess {idx}: Information Gained')
    axs[1,idx].set_xlabel("Entropy (bits)")
    axs[1,idx].set_ylabel("Probability")
    # axs[1,idx].grid(True, axis='both')

    uhist, ubins = np.histogram(move_data[:,1], bins=ibins, density=True)
    axs[0,idx].hist(ubins[:-1], ubins, weights=uhist*(ubins[1]-ubins[0]))
    axs[0,idx].set_title(f'Guess {idx}: Initial Uncertainty')
    axs[0,idx].set_xlabel("Entropy (bits)")
    axs[0,idx].set_ylabel("Probability")
    # axs[0,idx].grid(True, axis='both')



f.tight_layout()
plt.show()