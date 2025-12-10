import random
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Environment
# -----------------------------
env = gym.make("LunarLander-v3", render_mode=None)
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n

# -----------------------------
# Q-Network
# -----------------------------
class DQNLunarLander(nn.Module):
    def __init__(self, stateDim, actionDim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stateDim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, actionDim)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Hyperparameters
# -----------------------------
EPSILON = 1.0
EPSILON_DECAY = 1.005
MIN_EPSILON = 0.01
GAMMA = 0.99
REPLAY_BUFFER = []
BATCH_SIZE = 64
LR = 1e-3
TARGET_UPDATE_AFTER = 1000
LEARN_AFTER_STEPS = 4
MAX_TRANSITIONS = 100_000
NUM_EPISODES = 750
DEVICE = torch.device("cpu")

# -----------------------------
# Networks / Optimizer / Loss
# -----------------------------
qNet = DQNLunarLander(stateDim=stateDim, actionDim=actionDim).to(DEVICE)
targetNet = DQNLunarLander(stateDim=stateDim, actionDim=actionDim).to(DEVICE)
targetNet.load_state_dict(qNet.state_dict())
targetNet.eval()

optimizer = optim.Adam(qNet.parameters(), lr=LR)
lossFn = nn.HuberLoss()

# -----------------------------
# Replay Buffer helpers
# -----------------------------
def insertTransition(transition):
    if len(REPLAY_BUFFER) >= MAX_TRANSITIONS:
        REPLAY_BUFFER.pop(0)
    REPLAY_BUFFER.append(transition)

def sampleTransition(batchSize=BATCH_SIZE):
    batchSize = min(batchSize, len(REPLAY_BUFFER))
    indices = random.sample(range(len(REPLAY_BUFFER)), k=batchSize)
    sampled = [REPLAY_BUFFER[i] for i in indices]

    states, actions, rewards, nextStates, dones = zip(*sampled)
    states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(np.array(actions), dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=DEVICE)
    nextStates = torch.tensor(np.array(nextStates), dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(np.array(dones), dtype=torch.bool, device=DEVICE)
    return states, actions, rewards, nextStates, dones

# -----------------------------
# Epsilon-greedy policy
# -----------------------------
def policy(state, explore=0.0):
    if random.random() < explore:
        return random.randint(0, actionDim - 1)
    stateT = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        qValues = qNet(stateT)
        return int(torch.argmax(qValues, dim=1).item())

# -----------------------------
# Metric helpers
# -----------------------------
metrics = {"episode": [], "length": [], "totalReward": [], "avgQ": [], "exploration": []}
stepCounter = 0

# Gather random states for avg-Q metric
randomStates = []
obs, _ = env.reset()
for _ in range(20):
    randomStates.append(obs)
    a = policy(obs)
    nextObs, _, terminated, truncated, _ = env.step(a)
    obs = nextObs
    if terminated or truncated:
        obs, _ = env.reset()
randomStates = torch.tensor(np.array(randomStates), dtype=torch.float32, device=DEVICE)

def getQValues(states):
    with torch.no_grad():
        return qNet(states).max(1).values

# -----------------------------
# Training loop
# -----------------------------
for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    totalReward = 0.0
    episodeLength = 0

    while not done:
        action = policy(obs, explore=EPSILON)
        nextObs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        insertTransition((obs, action, reward, nextObs, done))
        obs = nextObs

        totalReward += reward
        episodeLength += 1
        stepCounter += 1

        # Learning step
        if len(REPLAY_BUFFER) >= BATCH_SIZE and (stepCounter % LEARN_AFTER_STEPS == 0):
            states, actions, rewards, nextStates, dones = sampleTransition(BATCH_SIZE)

            with torch.no_grad():
                nextQValues = targetNet(nextStates).max(1).values
                targets = rewards + GAMMA * nextQValues * (~dones)

            preds = qNet(states)
            currentQ = preds.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = lossFn(currentQ, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target update
        if stepCounter % TARGET_UPDATE_AFTER == 0:
            targetNet.load_state_dict(qNet.state_dict())

    # Epsilon decay
    EPSILON = max(MIN_EPSILON, EPSILON / EPSILON_DECAY)

    # Save metrics
    avgQ = getQValues(randomStates).mean().item()
    metrics["episode"].append(episode)
    metrics["length"].append(episodeLength)
    metrics["totalReward"].append(totalReward)
    metrics["avgQ"].append(avgQ)
    metrics["exploration"].append(EPSILON)

    if (episode + 1) % 5 == 0 or episode == 0:
        pd.DataFrame(metrics).to_csv("lunarlanderMetric.csv", index=False)
    print(f"Episode {episode+1:3d}/{NUM_EPISODES} | Reward: {totalReward:6.2f} | Eps: {EPSILON:5.3f}")

env.close()
torch.save(qNet.state_dict(), "DQNLunarlander.pt")
print("Training complete — Model saved as 'DQNLunarlander.pt'")
