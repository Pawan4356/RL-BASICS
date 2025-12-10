# Reinforcement Learning — Intro
### Concepts + Practical Implementation (CliffWalking Environment)

###  **Reinforcement Learning Fundamentals**

#### -> The ARO Framework
**A R O → Agent – Reward – Observation**

Every Reinforcement Learning (RL) problem consists of:
- **Agent** → Learns and acts
- **Environment** → The external system the agent interacts with
- **Reward** → Feedback signal that indicates success or failure



#### -> Types of Tasks
- **Episodic Task** → Has a definite start and end (e.g., game)
- **Continuing Task** → Never-ending (e.g., stock trading, industrial control)



#### -> Core RL Elements

| Concept | Description |
|-|--|
| **Environment** | The world the agent interacts with |
| **Reward** | Scalar feedback after each action |
| **Policy (π)** | Mapping from state to action |
| **Value (V)** | Expected long-term return |
| **Model** | Internal prediction of environment dynamics |



#### -> Value and Discount Factor

**Return:**  
**Gt = Rt+1 + γ * Rt+2 + γ^2 * Rt+3 + ...**
where **γ (0 < γ < 1)** controls how much future rewards are valued.



#### -> Environment Types
- **Fully Observable** → Complete state info available  
- **Partially Observable** → Noisy or incomplete info  
- **Deterministic** → Next state fixed by (S, A)  
- **Stochastic** → Next state has randomness  



###  **Markov Decision Process & RL Algorithms**

#### -> Markov Decision Process (MDP)
Mathematical framework for decision-making under uncertainty.  
Defined by the tuple:  
***(S, A, P, R, γ)***
- **S** → States  
- **A** → Actions  
- **P(s'|s,a)** → Transition probability  
- **R(s,a)** → Reward function  
- **γ** → Discount factor

**Markov Property:**  
Future depends only on the present, not the past.



#### -> Model-Based vs Model-Free RL

| Type | Description | Examples |
|--|--|--|
| **Model-Based** | Learns or uses environment model | Dynamic Programming, Dyna-Q |
| **Model-Free** | Learns directly from experience | Monte Carlo, TD, SARSA, Q-Learning |

#### -> Value Functions
- **State Value (Vπ(s))** → Expected return from state `s` under policy `π`  
- **Action Value (Qπ(s,a))** → Expected return from state-action pair under policy `π`



#### -> Exploration vs Exploitation
- **Exploration** → Try new actions for information  
- **Exploitation** → Use known best action for reward  

Common strategy: **ε-Greedy Policy**
> With probability ε → explore  
> With probability (1−ε) → exploit



#### -> Temporal Difference (TD) Learning
TD combines **Monte Carlo** + **Dynamic Programming** ideas.

**TD Update Rule:**

V(S-> t) ← V(S-> t) + α [R-> {t+1} + γV(S-> {t+1}) − V(S-> t)]



#### -> SARSA (On-Policy TD Control)

Q(S-> t,A-> t) ← Q(S-> t,A-> t) + α [R-> {t+1} + γQ(S-> {t+1},A-> {t+1}) − Q(S-> t,A-> t)]

Learns about the same policy it uses.



#### -> Q-Learning (Off-Policy TD Control)

Q(S-> t,A-> t) ← Q(S-> t,A-> t) + α [R-> {t+1} + γ \max-> a Q(S-> {t+1},a) − Q(S-> t,A-> t)]

Learns the optimal policy while following a different behavior.



##  Practical Implementation — CliffWalking Environment

This project applies the above theory using **OpenAI Gymnasium’s** `CliffWalking-v1` environment.

### Files Overview

| File | Description |
|--|--|
| `randomAgent.py` | Agent moves randomly (no learning). |
| `SARSA.py` | Implements the SARSA algorithm (on-policy TD control). |
| `Qlearning.py` | Implements the Q-Learning algorithm (off-policy TD control). |
| `evaluation.py` | Visualizes learned policy using the saved Q-table. |



## ⚙️ Setting up the Environment (VENV Creation)

Follow these steps carefully to create a clean environment for running your RL scripts.

### Step 1 — Create Virtual Environment
```bash
python3.10 -m venv venv
# Here, python 3.10 is used
````

### Step 2 — Activate the Virtual Environment

```bash
venv\Scripts\activate
```

### Step 3 — Install Required Packages

```bash
pip install numpy opencv-python gymnasium pygame
```
If it doesn't works, use requirements.txt. And note swig is installed externally...Be carefulllll!


## How to Run Each Program

### 1. Random Agent

```bash
python randomAgent.py
```

* Agent moves randomly in the environment.
* No learning occurs.

### 2. Train SARSA Agent

```bash
python SARSA.py
```

* Trains SARSA for 500 episodes.(Adjustable)
* Saves Q-table as `SarsaQTable.pkl`.

### 3. Train Q-Learning Agent

```bash
python Qlearning.py
```

* Trains Q-learning for 1000 episodes.(Adjustable)
* Saves Q-table as `QLearningTable.pkl`.

### 4. Evaluate Learned Agent

```bash
python evaluation.py
```

* Loads and visualizes agent performance.
* Edit this line in the script to switch between SARSA and Q-Learning:

  ```python
  # qTable = pkl.load(open("SarsaQTable.pkl", "rb"))
  qTable = pkl.load(open("QLearningTable.pkl", "rb"))
  ```



### Visualization

* **A** → Agent
* **G** → Goal
* **CLIFF** → Dangerous region (negative reward)

Goal → Learn to reach **G** safely avoiding the **cliff**.


### References

* [Reinforcement Learning INtro.](https://youtu.be/zdIQkjtFX_I?si=slQA3ODrVy3v_Rp9)
* [How do RL Agents Learn?](https://youtu.be/DLcBjo5gIxs?si=0M8ou1fs68wfHBWr)
* [How to Train your first RL Agent.](https://youtu.be/tbpBW5Yr44k?si=kI7AVOrtPD7VGacZ)
* [Gymnasium Documentation](https://gymnasium.farama.org/)


**Author - Pawankumar Navinchandra**