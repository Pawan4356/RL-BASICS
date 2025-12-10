'''
Implementing SARSA Algorithm to improve the iterations!
'''

import pickle
import numpy as np
import gymnasium as gym

cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")
qTable = np.zeros(shape=(48, 4))

# Parameters
EPSILON = 0.05
ALPHA = 0.1
GAMMA = 0.9
NUMEPISODES = 500

def policy(state, explore=0.0):
    actionStep = int(np.argmax(qTable[state]))
    if np.random.rand() <= explore:
        actionStep = int(np.random.randint(low=0, high=4))
    return actionStep

for episode in range(NUMEPISODES):
    totalReward = 0
    episodeLength = 0
    done = False
    state, _ = cliffEnv.reset()
    action = policy(state, EPSILON)

    while not done:
        nextState, reward, done, _, info = cliffEnv.step(action)
        nextAction = policy(nextState, EPSILON)

        qTable[state][action]  += ALPHA * (reward + (GAMMA * qTable[nextState][nextAction]) - qTable[state][action])
        state = nextState
        action = nextAction

        episodeLength += 1
        totalReward += reward

    print(f"Episode: {episode+1:5d} | Episode Length: {episodeLength:3d} | Total Reward: {totalReward:3d}")

# np.savetxt("Q Table.csv", qTable, delimiter=",")
with open("SarsaQTable.pkl", "wb") as f:
    pickle.dump(qTable, f)

cliffEnv.close()