import gym
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

def linear_state(state):
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)

with open('MountainCar-v0-q-learning.pickle', 'rb') as f:
    Q = pickle.load(f)
    print("Model loaded!")

scores = []
env = gym.make('MountainCar-v0')


for _ in range(100):
    score = 0
    state = env.reset()
    while True:
        env.render()
        time.sleep(0.01)
        state = linear_state(state)
        action = np.argmax(Q[state]) if state in Q else np.random.choice([0, 1, 2])
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            scores.append(score)
            print(f'score: {score}')
            break
env.close()
print(f'Average score: {sum(scores)/len(scores)}')
plt.figure(2, figsize=[10,5])
plt.plot(scores, 'b.', alpha=0.8)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('100 Trials Test Scores')
plt.savefig('Test Scores.png')
plt.show()