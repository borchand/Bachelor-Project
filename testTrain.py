import gymnasium as gym
import numpy as np 
import pickle
from tqdm import tqdm
# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v1')
Q = np.zeros([env.observation_space.n,env.action_space.n])
# env.observation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-learning
eta = .628
gma = .9
epis = 500
rev_list = [] # rewards per episode calculate
# 3. Q-learning Algorithm
for i in tqdm(range(epis)):
    # Reset environment
    s = env.reset()[0]
    rAll = 0
    d = False
    j = 0
    d = False
    t = False
    # The Q-Table learning algorithm
    while not d or not t:
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        s1, r, d, t, info = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        s = s1
        # Code will stop at d == True, and render one state before it
    rev_list.append(rAll)
    
env.close()
print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)
pickle.dump(Q, open("frozenLake_qTable.pkl", "wb"))