from matplotlib import pyplot as plt
import numpy as np

NUM_BINS=10

data = np.genfromtxt('entropy.csv', delimiter=',')
# data[sample] = (guess_num in <1..>, uncertainty, information)

# datanz = data[data[:,1] != 0]

moves = np.unique(data[:,0])
moves.sort()
f, axs = plt.subplots(2, len(moves), sharex='col')
f.set_size_inches(10, 5.5)

stats = np.zeros((3, 5))
print(moves)

for idx, move in enumerate(moves):
    stats[idx, 0] = move
    print(f'-- Guess {move:1.0f} --')
    move_data = data[data[:,0] == move, :]
    ihist, ibins = np.histogram(move_data[:,2], bins=NUM_BINS, density=True)
    axs[1,idx].hist(ibins[:-1], ibins, weights=ihist*(ibins[1]-ibins[0]), edgecolor='k')
    axs[1,idx].set_title(f'Guess {idx}: Information Gained')
    axs[1,idx].set_xlabel("Entropy (bits)")
    axs[1,idx].set_ylabel("Probability")

    stats[idx, 3] = np.mean(-move_data[:,2])
    stats[idx, 4] = np.std(-move_data[:,2])
    print(f'\tI Gain:      mean: {stats[idx,3]:5.4f}, std: {stats[idx,4]:5.4f}')

    # axs[1,idx].grid(True, axis='both')

    uhist, ubins = np.histogram(move_data[:,1], bins=ibins, density=True)
    axs[0,idx].hist(ubins[:-1], ubins, weights=uhist*(ubins[1]-ubins[0]), edgecolor='k')
    axs[0,idx].set_title(f'Guess {idx}: Initial Uncertainty')
    axs[0,idx].set_xlabel("Entropy (bits)")
    axs[0,idx].set_ylabel("Probability")
    axs[0,idx].tick_params(labelbottom=True)

    stats[idx, 1] = np.mean(move_data[:,1])
    stats[idx, 2] = np.std(move_data[:,1])
    print(f'\tUncertainty: mean: {stats[idx,1]:5.4f}, std: {stats[idx,2]:5.4f}')



f.tight_layout()
plt.show()

with open('entropy_stats.csv', 'w') as file:
    print(f'guess,uncmean,uncstd,gainmean,gainstd', file=file)
    for row in stats:
        print(','.join(str(item) for item in row), file=file)