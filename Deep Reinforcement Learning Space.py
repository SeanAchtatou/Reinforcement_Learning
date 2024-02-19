import gym
import cv2
import time
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory




def no_DRL(env):

    for i in range(10):
        epoches = 0
        rewards = 0
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, info, _ = env.step(action)

            rewards += reward
            img = env.render()
            cv2.imshow("Running of the Agent in Environment untrained",img)


            #print(f"{0:_^10}")
            #print(f"State: {state}")
            #print(f"Action: {action}")
            #print(f"Reward: {reward}")
            #print(f"Epoches: {epoches}")
            cv2.waitKey(1)
            time.sleep(0.1)
            epoches += 1

        print(f"{f'Episode: {i}':-^32}")
        print(f"Steps: {epoches}")
        print(f"Rewards: {rewards}")

    cv2.destroyAllWindows()


def DRL(env):
    shape = env.observation_space.shape[0]
    actions = env.action_space.n                             #Our states of actions in the Environment

    model = Sequential()
    model.add(Flatten(input_shape=(1,shape)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions,activation="linear"))

    agent = DQNAgent(
        model=model,
        memory=SequentialMemory(limit=1000,window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=50,
        target_model_update=0.05
    )

    agent.compile(Adam(lr=0.001),metrics=["mae"])
    agent.fit(
        env,
        nb_steps=100000,
        visualize=True,
        verbose=1
    )


    #agent.test(env,nb_episodes= 10,visualize=True)

    return agent


def DRL_Load(env):
    shape = env.observation_space.shape[0]
    actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,shape)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions,activation="linear"))

    agent = DQNAgent(
        model=model,
        memory=SequentialMemory(limit=1000,window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=0.05
    )
    agent.compile(Adam(lr=0.001),metrics=["mae"])

    agent.load_weights("SPACE-V2_weights.h5f")
    agent.test(env,nb_episodes=10,visualize=True)



if __name__ == "__main__":
    environment = ["LunarLander-v2"]
    envG = gym.make(environment[0],render_mode="rgb_array").env

    if True:
        envG.reset()                                             #Reset the Environment
        cv2.imshow("Image of the Agent in Envrironment",envG.render())
        cv2.waitKey(0)
        actions = envG.action_space                              #Our states of actions in the Environment
        states = envG.observation_space                          #Our possibles states in the Environment
        #print(actions)
        #print(states)

        #no_DRL(envG)      #No training
        model = DRL(envG) #Training
        model.save_weights("SPACE-V2_weights.h5f",overwrite=True)
        #DRL_Load(envG)    #Re-use of model weights
