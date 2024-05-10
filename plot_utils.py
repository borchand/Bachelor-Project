import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

import pandas as pd
sns.set_theme(rc={"figure.dpi":100, 'savefig.dpi':6000})
plt.style.use('seaborn-whitegrid')

def data_dict(methods, envs, seeds, folder="results"):
    data = {}

    for method in methods:
        data[method] = {}
        for env in envs:
            data[method][env] = {}
            for seed in seeds:
                data[method][env][seed] = load_data(method, env, seed, folder=folder)
    return data

def load_data(method, env, seed, folder='results'):
    path = f"{folder}/{method}/{env}/{method}_{seed}.csv"
    data = pd.read_csv(path)
    # add column for accumulated reward
    data['accumulated reward'] = data['reward'].cumsum()

    # add column for success rate
    data['success rate'] = data['success'].cumsum() / (data.index + 1)

    return data

def create_plot_data(data, method, env, seeds, ax):
    plot_data = pd.DataFrame()

    for seed in seeds:
        plot_data = pd.concat([plot_data, data[method][env][seed][[ax]]], axis=1)
        # mean of accumulated reward
    mean = plot_data.mean(axis=1)
    std = plot_data.std(axis=1)
    
    plot_data["mean"] = mean
    plot_data["std"] = std
    plot_data["episode"] = plot_data.index
    return plot_data

def plot(data, methods, env, seeds, ax, p=plt, xPos=0):
    
    for method in methods:
        plot_data = create_plot_data(data, method, env, seeds, ax)
        p.plot(plot_data["episode"], plot_data["mean"], label=method)
        p.fill_between(plot_data["episode"], plot_data["mean"] - plot_data["std"], plot_data["mean"] + plot_data["std"], alpha=0.2)
        # add legend
        # p.legend()

    # p.ticklabel_format(axis='both', style='scientific', scilimits=(-10,10))



    # if plt is the default plt
    if p == plt:
        p.xlabel("Episode")
        p.ylabel(ax[0].upper() + ax[1:])
        p.title(f"{env}")
        # if success rate, set y limit to [0,1]
        if ax == "success rate":
            p.ylim(0, 1)
    else:
        p.set_xlabel("Episode")
        if xPos == 0:
            p.set_ylabel(ax[0].upper() + ax[1:])
        p.set_title(f"{env}")
        # if success rate, set y limit to [0,1]
        if ax == "success rate":
            p.set_ylim(0, 1)

def create_plot_grid(data, methods, envs, seeds, ax, save_name=None):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    # add margin bewteen subplots
    fig.subplots_adjust(hspace = 0.5, wspace=.25)

    # add title to the whole plot
    # fig.suptitle("Comparing abstraction methods on " + ax)

    for i, env in enumerate(envs):
        plot(data, methods, env, seeds, ax, axs[i // 3][i % 3], xPos=i%3)
    
    handles, labels = axs[i // 3][i % 3].get_legend_handles_labels()
    axs[1][1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=3)
    #save the plot as svg
    if save_name is not None:
        plt.savefig(f"images/{save_name}-{ax}.pdf", bbox_inches='tight', format='pdf')
    else:
        plt.savefig(f"images/{ax}.pdf", bbox_inches='tight', format='pdf')
    plt.show()
