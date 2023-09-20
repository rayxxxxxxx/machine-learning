import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

import vars
from modules.neuron_chain import NeuronChain
from modules.neuron_chain_v2 import NeuronChainV2


def main():
    model1 = NeuronChain(vars.N, vars.STIMULI_EVAPORATION_COEFF,
                         vars.THRESHOLD_EVAPORATION_COEFF)

    model2 = NeuronChainV2(vars.N, vars.STIMULI_EVAPORATION_COEFF,
                           vars.THRESHOLD_EVAPORATION_COEFF)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    ax = fig.add_subplot()
    twinax = ax.twinx()

    S = []
    T = 1000
    for i in range(T):
        S.append(model1.s.copy())
        # S.append(model2.s.copy())

        # x = np.ones(vars.N)*0.01
        # x = np.linspace(1, vars.N, vars.N)*0.01
        # x = np.sin(0.01*(np.linspace(1, vars.N, vars.N)))

        x = 0.1
        # x = 1.0 if i % 250 == 0 else 0
        # x = 0.1 if i < T//4 else (0.5 if i < T//2 else 1.0)
        # x = math.sin(0.0001*i)

        model1.update(x)
        # model2.update(x)

    S = np.array(S).transpose()

    color = iter(cm.rainbow(np.linspace(0, 1, vars.N)))
    for i, s in enumerate(S[1:]):
        ax.plot(range(T), s, label=f'{i+1}', lw=1, c=next(color))
    twinax.plot(range(T), S[0], lw=1, ls='--', c='black', label=f'0')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
