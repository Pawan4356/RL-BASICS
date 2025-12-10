import cv2
import numpy as np
import pickle as pkl
import gymnasium as gym

NUMEPISODES = 1

cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")
qTable = pkl.load(open("SarsaQTable.pkl", "rb"))
# qTable = pkl.load(open("QLearningTable.pkl", "rb"))
width, height = 600, 210
boxWidth = int(width/12 - 1)

# Handy functions for Visuals
def initializeFrame():
    img = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255
    margin_horizontal = 6
    margin_vertical = 6

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (boxWidth * i + margin_horizontal, margin_vertical),
                       (boxWidth * i + margin_horizontal, height - margin_vertical), 
                       color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, boxWidth * i + margin_vertical),
                       (600 - margin_horizontal, boxWidth * i + margin_vertical), 
                       color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (boxWidth * 1 + margin_horizontal + 2, boxWidth * 3 + margin_vertical + 2),
                        (boxWidth * 11 + margin_horizontal - 2, boxWidth * 4 + margin_vertical - 2), 
                        color=(0, 42, 42), thickness=-1)
    
    img = cv2.putText(img, text="CLIFF", 
                      org=(boxWidth * 5 + margin_horizontal, boxWidth * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                      fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", 
                        org=(boxWidth * 11 + margin_horizontal + 10, boxWidth * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=(0, 0, 0), thickness=2)
    # Start
    # frame = cv2.putText(img, text="S", 
    #                     org=(boxWidth * 0 + margin_horizontal + 10, boxWidth * 4 + margin_vertical - 10),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
    #                     fontScale=1, color=(0, 0, 0), thickness=2)
    return frame

def putAgent(img, state):
    margin_horizontal = 6
    margin_vertical = 6
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", 
                org=(boxWidth * column + margin_horizontal + 10, boxWidth * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, color=(0, 0, 0), thickness=2)
    return img

def policy(state, explore=0.0):
    actionStep = int(np.argmax(qTable[state]))
    if np.random.rand() <= explore:
        actionStep = int(np.random.randint(low=0, high=4))
    return actionStep

for episode in range(NUMEPISODES):
    done = False
    state, _ = cliffEnv.reset()
    frame = initializeFrame()

    while not done:
        tempFrame = putAgent(frame.copy(), state=state)
        cv2.imshow("CLiff Walking", tempFrame)
        cv2.waitKey(100)
        action = policy(state)
        # action = cliffEnv.action_space.sample()
        state, reward, done, _,info = cliffEnv.step(action)

cliffEnv.close()