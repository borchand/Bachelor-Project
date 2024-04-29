import os
import pickle
import pandas as pd


def save_log(log_data, log_info):
    """
    Save the log data
    """

    # create folder results if it does not exist
    if not os.path.exists("results/"):
        os.makedirs("results/")

    # create folder results/agent if it does not exist
    if not os.path.exists("results/" + log_info["agent"][0]):
        os.makedirs("results/" + log_info["agent"][0])

    # create folder results/agent/env if it does not exist
    if not os.path.exists("results/" + log_info["agent"][0] + "/" + log_info["env"][0]):
        os.makedirs("results/" + log_info["agent"][0] + "/" + log_info["env"][0])

    df = pd.DataFrame(log_data)

    df.to_csv("results/" + log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0]) + ".csv")

    print(log_info)
    df_info = pd.DataFrame(log_info)

    df_info.to_csv("results/" + log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0]) + "_info.csv")

def save_model(agent, log_info):
    """
    Save the agent
    """

    # create folder models if it does not exist
    if not os.path.exists("models/"):
        os.makedirs("models/")

    # create folder models/agent if it does not exist
    if not os.path.exists("models/" + log_info["agent"][0]):
        os.makedirs("models/" + log_info["agent"][0])

    # create folder models/agent/env if it does not exist
    if not os.path.exists("models/" + log_info["agent"][0] + "/" + log_info["env"][0]):
        os.makedirs("models/" + log_info["agent"][0] + "/" + log_info["env"][0])

    file_name = log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0])

    # save the agent and abstraction
    pickle.dump(agent, open("models/" + file_name + "_agent.pkl", "wb"))

def save_abstraction(abstraction, log_info):
    """
    Save the abstraction
    """

    # create folder models if it does not exist
    if not os.path.exists("models/"):
        os.makedirs("models/")

    # create folder models/agent if it does not exist
    if not os.path.exists("models/" + log_info["agent"][0]):
        os.makedirs("models/" + log_info["agent"][0])

    # create folder models/agent/env if it does not exist
    if not os.path.exists("models/" + log_info["agent"][0] + "/" + log_info["env"][0]):
        os.makedirs("models/" + log_info["agent"][0] + "/" + log_info["env"][0])

    file_name = log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0])

    # save the agent and abstraction
    pickle.dump(abstraction, open("models/" + file_name + "_abs.pkl", "wb"))

def load_model(agent_name, env, seed):
    """
    Load the agent
    """

    file_name = agent_name + "/" + env + "/" + agent_name + "_" + str(seed)

    # load the agent and abstraction
    agent = pickle.load(open("models/" + file_name + "_agent.pkl", "rb"))
    return agent

def load_abstraction(agent_name, env, seed):
    """
    Load the agent and abstraction  (mainly for CAT-RL)
    """
    file_name = agent_name + "/" + env + "/" + agent_name + "_" + str(seed)

    # load the agent and abstraction
    abstraction = pickle.load(open("models/" + file_name + "_abs.pkl", "rb"))
    return abstraction