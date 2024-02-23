import gymnasium as gym
import numpy as np 
import pickle
# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v1', render_mode="human")
Q = pickle.load(open("frozenLake_qTable.pkl", "rb"))
# env.observation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-learning
eta = .628
gma = .9
epis = 5000
rev_list = [] # rewards per episode calculate
# 3. Q-learning Algorithm
for i in range(epis):
    # Reset environment
    s = env.reset()[0]
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        s1,r,d,_,_= env.step(a)

        s = s1
        if d == True:
            break
    rev_list.append(rAll)
env.render()    
env.close()