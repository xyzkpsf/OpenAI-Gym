import gym
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# provides a default value for the key that does not exists.
Q = defaultdict(lambda: [0, 0, 0])
VISITED = defaultdict(lambda: [0, 0, 0])

env = gym.make("MountainCar-v0")

# Map state space into a linear space
def linear_state(state):
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)

discount = 0.95
EPOCHS = 50000
score_list = []
exploration_decay_rate = 4
for epoch in range(EPOCHS):
    # take some linear function map lr from 0.75 to about 0.01
    lr = 0.75 - (epoch/EPOCHS)/1.35
    state = linear_state(env.reset())
    score = 0
    exploration_rate = 0.01 + (1-0.01)*np.exp(-exploration_decay_rate * epoch / EPOCHS)
    #max_position = state[0]
    while True:
        action = np.argmax(Q[state])
        if np.random.random() <= exploration_rate:
            action = np.random.choice([0, 1, 2]) 

        next_s, reward, done, _ = env.step(action)
        if next_s[0] > 0.4:
            reward += 1
        next_s = linear_state(next_s)
        Q[state][action] = (1 - lr) * Q[state][action] + lr * (reward + discount * max(Q[next_s]))
        score += reward
        state = next_s

        if done:
            score_list.append(score)
            if epoch % 2000 == 0:
                print('episode:', epoch, '; Aver last 2000 score:', np.average((score_list[-2000:])), '; max scores:', max((score_list[-2000:])))
            break
            
    if np.average((score_list[-1000:])) > -100:
        plt.plot(score_list)
        break
    epoch += 1
        
env.close()

with open('MountainCar-v0-q-learning.pickle', 'wb') as f:
    pickle.dump(dict(Q), f)
    print('model saved')
plt.figure(2, figsize=[10,5])
plt.plot(score_list, 'b.', alpha=0.8)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('FInal Training Scores')
plt.savefig('FInal Training Scores.png')
plt.show()