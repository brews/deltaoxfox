import matplotlib.pylab as plt


def predictplot(prediction, ylabel=None, x=None, xlabel=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if x is None:
        x = list(range(len(prediction.ensemble)))

    perc = prediction.percentile(q=[5, 50, 95])

    ax.fill_between(x, perc[:, 0], perc[:, 2], alpha=0.25,
                    label='90% uncertainty', color='C0')

    ax.plot(x, perc[:, 1], label='Median', color='C0')
    ax.plot(x, perc[:, 1], marker='.', color='C0')

    # if prediction.prior_mean is not None:
    #     ax.axhline(prediction.prior_mean, label='Prior mean',
    #                linestyle='dashed', color='C1')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax
