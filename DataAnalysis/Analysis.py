import pickle
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns


def getAgentEpisodeData(data, algorithm, numAgents, environment):
    agents = defaultdict(lambda: defaultdict(lambda: []))
    for episode, steps in data["algorithms"][algorithm][numAgents][environment].items():
        if episode != "parameters":
            for stepNumber, step in steps.items():
                for agent, data in step.items():
                    if agent != "centralLearner":
                        agents[agent][episode].append(data)

    return {agent: [data for data in episode.values()] for agent, episode in agents.items()}

def getCLEpisodeData(data, algorithm, numAgents, environment):
    CLSteps = []
    for episode, steps in data["algorithms"][algorithm][numAgents][environment].items():
        if episode != "parameters":
            for stepNumber, step in steps.items():
                CLSteps.append(step["centralLearner"])

    return CLSteps

def getAgentEpisodeReward(data, algorithm, numAgents, environment):
    episodeData = getAgentEpisodeData(data, algorithm, numAgents, environment)
    return {agent: [sum([step["reward"] for step in steps]) for steps in episode] for agent, episode in episodeData.items()}

def getAlgorithmEpisodeAverageReward(data, algorithm, numAgents, environment):
    data = getAgentEpisodeReward(data, algorithm, numAgents, environment)
    return np.mean(np.array(list(data.values())), axis=0)

def getAgentEpisodeNumMessages(data, algorithm, numAgents, environment):
    episodeData = getAgentEpisodeData(data, algorithm, numAgents, environment)
    return {agent: [sum([step["?message"] is not None for step in steps]) for steps in episode] for agent, episode in episodeData.items()}

def getAgentEpisodeAverageNumMessages(data, algorithm, numAgents, environment):
    data = getAgentEpisodeNumMessages(data, algorithm, numAgents, environment)
    return np.mean(np.array(list(data.values())), axis=0)

def getAgentEpisodeAverageNumberOfMessagesPerStep(data, algorithm, numAgents, environment):
    episodeData = getAgentEpisodeData(data, algorithm, numAgents, environment)
    return {agent: [np.mean([step["?message"] is not None for step in steps]) for steps in episode] for agent, episode in episodeData.items()}

def getAgentEpisodeSizeMessages(data, algorithm, numAgents, environment):
    episodeData = getAgentEpisodeData(data, algorithm, numAgents, environment)
    return {agent: [sum([sys.getsizeof(message) if (message := step["?message"]) is not None else 0 for step in steps]) for steps in episode] for agent, episode in episodeData.items()}

def getFinalValueFunction(data, algorithm, numAgents, environment):
    episodeData = getCLEpisodeData(data, algorithm, numAgents, environment)
    return episodeData[-1]["?valueFunction"]

def getVfromQ(q):
    tempv = {state: np.max(actions) for state, actions in q.items()}
    v = []
    for state, action in tempv.items():
        x = state[0]
        y = state[1]
        while len(v) - 1 < x:
            v.append([])
        while len(v[x]) - 1 < y:
            v[x].append(None)
        v[x][y] =  action
    return v

def printV2D(q):
    text = "V:\n"
    v = getVfromQ(q)
    for row in v:
        for entry in row:
            text += "{:.2f} ".format(entry)
        text += "\n"
    print(text)

def plotAgentEpisodeReward(data, algorithm, numAgents, environment):
    agentRewards = getAgentEpisodeReward(data, algorithm, numAgents, environment)
    for agent, y in agentRewards.items():
        plt.plot([x for x in range(len(y))],y, label=agent)    
    plt.xlabel('Episode number')
    plt.ylabel('Total reward')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def plotAlgorithmEpisodeAverageReward(data, algorithms, numAgents, environment):
    algorithmRewards = {algorithm: getAlgorithmEpisodeAverageReward(data, algorithm, numAgents, environment) for algorithm in algorithms}
    for algorithm, y in algorithmRewards.items():
        plt.plot([x for x in range(len(y))],y, label=algorithm)    
    plt.xlabel('Episode number')
    plt.ylabel('Average agent reward')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def plotAgentEpisodeNumMessages(data, algorithm, numAgents, environment):
    agentMessages = getAgentEpisodeNumMessages(data, algorithm, numAgents, environment)
    for agent, y in agentMessages.items():
        plt.plot([x for x in range(len(y))],y, label=agent)    
    plt.xlabel('Episode number')
    plt.ylabel('Number of messages')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def plotAlgorithmEpisodeAverageNumMessages(data, algorithms, numAgents, environment):
    algorithmMessages = {algorithm: getAgentEpisodeAverageNumMessages(data, algorithm, numAgents, environment) for algorithm in algorithms}
    for algorithm, y in algorithmMessages.items():
        plt.yscale("log")    
        plt.plot([x for x in range(len(y))],y, label=algorithm) 
    plt.xlabel('Episode number')
    plt.ylabel('Average agent number of messages')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def plotAgentEpisodeAverageNumberOfMessagesPerStep(data, algorithm, numAgents, environment):
    agentMessages = getAgentEpisodeAverageNumberOfMessagesPerStep(data, algorithm, numAgents, environment)
    for agent, y in agentMessages.items():
        plt.plot([x for x in range(len(y))],y, label=agent)    
    plt.xlabel('Episode number')
    plt.ylabel('Average Number of messages')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def create_grids(q, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in q.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig
