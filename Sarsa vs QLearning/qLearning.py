import numpy as np
import pickle as pkl
import gymnasium as gym

cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")
qTable = np.zeros(shape=(48, 4))

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUMEPISODES = 1000

def policy(state, explore=0.0):
    actionStep = int(np.argmax(qTable[state]))
    if np.random.rand() <= explore:
        actionStep = int(np.random.randint(low=0, high=4))
    return actionStep

for episode in range(NUMEPISODES):
    done = False
    totalReward = 0
    episodeLength = 0
    state, _ = cliffEnv.reset()
    while not done:
        action = policy(state, EPSILON)
        nextState, reward, done, _, info = cliffEnv.step(action)
        nextAction = policy(nextState)
        qTable[state][action] += ALPHA * ((reward + GAMMA * np.max(qTable[nextState])) - qTable[state][action])
        state = nextState
        totalReward += reward
        episodeLength += 1
    
    print(f"Episode: {episode:5d} | Episode Length: {episodeLength:5d} | Total Reward: {totalReward:5d}")

cliffEnv.close()
pkl.dump(qTable, open("QLearningTable.pkl", "wb"))