'''
Without any Learning or Rewarding function,
Just Random movements to reach the goal!
'''

import cv2
import numpy as np
import gymnasium as gym

cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")
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

done = False
frame = initializeFrame()
state, info = cliffEnv.reset()

while not done:
    # print(cliffEnv.render())
    tempFrame = putAgent(frame.copy(), state)
    cv2.imshow("CliffWalking", tempFrame)
    cv2.waitKey(50)
    action = int(np.random.randint(low=0, high=4))
    # print(state, "-->", ["up", "right", "down", "left"][action])
    state, reward, done, _, info = cliffEnv.step(action)