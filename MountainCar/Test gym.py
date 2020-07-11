import gym
import random
import time
from collections import defaultdict


env = gym.make("MountainCar-v0")
env.reset()
score = 0

print(env.observation_space.low)
print(env.observation_space.high)

for _ in range(2):
    while True:
        env.render()
        time.sleep(0.01)
        # Random move
        nextMove = random.randint(0,2)
        # Return of next move
        state, reward, done, info = env.step(nextMove)
        score += reward
        if done:
            print(f'score: {score}')
            break
    env.close()