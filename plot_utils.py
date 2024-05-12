import seaborn as sns
import matplotlib.pyplot as plt
from run_icml import split_max_episodes

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

import pandas as pd
sns.set_theme(rc={"figure.dpi":100, 'savefig.dpi':6000})
plt.style.use('seaborn-whitegrid')


CartPoleEpisodes = 6000
AcrobotEpisodes = 2000
MountainCarEpisodes = 5000
MountainCarContinuousEpisodes = 1000
LunarLanderEpisodes = 6000
PendulumEpisodes = 6000

PENDULUM_K = 25
MOUNTAINCARCONTINUOUS_K = 2

# envs
max_episodes_dict = {
    "Acrobot-v1": AcrobotEpisodes,
    "CartPole-v1": CartPoleEpisodes,
    "MountainCar-v0": MountainCarEpisodes,
    "MountainCarContinuous-v0": MountainCarContinuousEpisodes,
    "Pendulum-v1": PendulumEpisodes,
    "LunarLander-v2": LunarLanderEpisodes
}

old_splits_dict = {
    "Acrobot-v1": False,
    "CartPole-v1": False,
    "MountainCar-v0": False,
    "MountainCarContinuous-v0": False,
    "Pendulum-v1": False,
    "LunarLander-v2": False
}

def data_dict(methods, envs, seeds, folder="results"):
    data = {}

    for method in methods:
        data[method] = {}
        for env in envs:
            data[method][env] = {}
            for seed in seeds:
                data[method][env][seed] = load_data(method, env, seed, folder=folder)
    return data

def data_dict_icml(algos: list[str], envs: list[str], seeds: list[str]):
    
    data = {}
    for algo in algos:
        data[algo] = {}
        for env in envs:
            data[algo][env] = {}
            for seed in seeds:
                data[algo][env][seed] = load_data_icml(env, algo, seed, max_episodes_dict[env], old_splits_dict[env])

    return data

def load_data(method, env, seed, folder='results'):
    path = f"{folder}/{method}/{env}/{method}_{seed}.csv"
    data = pd.read_csv(path)
    # add column for accumulated reward
    data['accumulated reward'] = data['reward'].cumsum()

    # add column for success rate
    data['success rate'] = data['success'].cumsum() / (data.index + 1)

    return data

def load_data_icml(env: str, algo: str, seed: int, max_episodes: int, old_split=False) -> pd.DataFrame:

    policy_episodes_percent = 0.8 if old_split else 0.6
    policy_episodes, experiment_episodes = split_max_episodes(max_episodes, policy_episode_percent=policy_episodes_percent)
    
    if env == "MountainCarContinuous-v0":
        path = f"results/icml/{env}/icml_{policy_episodes}_{MOUNTAINCARCONTINUOUS_K}_{algo}_{experiment_episodes}_{seed}.csv"
    elif env == "Pendulum-v1":
        path = f"results/icml/{env}/icml_{policy_episodes}_{PENDULUM_K}_{algo}_{experiment_episodes}_{seed}.csv"
    else:
        path = f"results/icml/{env}/icml_{policy_episodes}_{algo}_{experiment_episodes}_{seed}.csv"

    data = pd.read_csv(path)
    # add column for accumulated reward
    data['accumulated reward'] = data['rewards'].cumsum()

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

def create_plot_data_icml(data, method, env, seeds, ax):
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
        if method == "binQ":
            method = "Discretization"
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

def plot_icml(data, algos: list[str], env: str, seeds:int, ax: str, p=plt, xPos:int =0, colors: list[str]=["red"]):
    
    for algo in algos:
        plot_data = create_plot_data_icml(data, algo, env, seeds, ax)
        p.plot(plot_data["episode"], plot_data["mean"], label="Demonstrator", color=colors[0])
        p.fill_between(plot_data["episode"], plot_data["mean"] - plot_data["std"], plot_data["mean"] + plot_data["std"], alpha=0.2, color=colors[0])
        # add legend
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

def create_plot_grid_icml(data, algos: list[str], envs: list[str], seeds: list[int], ax: str, save_name="ppo"):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    # add margin bewteen subplots
    fig.subplots_adjust(hspace = 0.5, wspace=.4)

    # add title to the whole plot
    # fig.suptitle("Comparing abstraction methods on " + ax)

    for i, env in enumerate(envs):
        plot_icml(data, algos, env, seeds, ax, axs[i // 3][i % 3], xPos=i%3)
    
    handles, labels = axs[i // 3][i % 3].get_legend_handles_labels()
    axs[1][1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=3)
    #save the plot as svg
    if save_name is not None:
        plt.savefig(f"images/{save_name}-{ax}.pdf", bbox_inches='tight', format='pdf')
    else:
        plt.savefig(f"images/{ax}.pdf", bbox_inches='tight', format='pdf')
    plt.show()

