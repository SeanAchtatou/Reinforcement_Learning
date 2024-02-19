import gym
import cv2
import time
import random
import numpy as np

def change_State(env):
    state = env.encode(3,1,5,0)  #row (y) taxi location, column (x) taxi location, passenger location, passenger dropoff
    print(f'State : {state}')
    env.s = state   #Put new state in the environment
    image = env.render()
    print(env.P[state])
    cv2.imshow("Im",image)
    cv2.waitKey(0)


def QRL(env):
    Q_table = np.zeros([env.observation_space.n,env.action_space.n])

    alpha = 0.3     #Learning rate
    gamma = 0.6      #Importance of future reward
    epsilon = 0.2  #Exploration or best action
    nb_generations = 10001

    for i in range(nb_generations):
        state = env.reset()[0]
        epoches = 0
        penalties = 0
        done = False

        while not done:
            if random.uniform(0,1) < epsilon:      #Exploration
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])       #Best action

            next_state, reward, done, info, _ = env.step(action)    #Take action
            old_reward = Q_table[state,action]       #Get reward for the action in the actual state
            next_reward = np.max(Q_table[next_state])   #Get reward for the best action in the next state

            Q_learning_state = ((1-alpha) * old_reward) + (alpha * (reward + gamma * (next_reward)))  #Calculate the best new reward for actual action
            Q_table[state,action] = Q_learning_state

            if reward == -10:
                penalties += -1

            state = next_state
            epoches += 1

        if i%1000 == 0:
            print(f"Generation: {i}")
            print(f"Penalties: {penalties}")

    return Q_table


def run_Agent(env,Q_table):
    episodes = 100
    total_penalties, total_epoches = 0,0

    for i in range(episodes):
        state = env.reset()[0]
        penalties, reward, epoches = 0,0,0

        done = False
        while not done:
            action = np.argmax(Q_table[state])
            state, reward, done, info, _ = env.step(action)

            if reward == -10:
                penalties += -1

            img = env.render()
            cv2.putText(img, f"Environment State: {str(state)}", (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
            cv2.putText(img, f"Action: {str(action)}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
            cv2.imshow("Running of the Agent in Environment trained",img)
            #print(f"{0:_^10}")
            #print(f"State: {state}")
            #print(f"Action: {action}")
            #print(f"Reward: {reward}")
            #print(f"Epoches: {epoches}")
            cv2.waitKey(1)
            time.sleep(0.2)


            epoches += 1

        print(f"Episode: {i}")
        print(f"Epoches: {epoches}")
        print(f"Penalties: {penalties}")
        total_epoches += epoches
        total_penalties += penalties


def no_Rl(env):
    epoches = 0
    penalties, reward = 0,0

    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info, _ = env.step(action)

        if reward == -10:
            penalties += -1

        img = env.render()
        cv2.putText(img, f"Environment State: {str(state)}", (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
        cv2.putText(img, f"Action: {str(action)}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
        cv2.imshow("Running of the Agent in Environment untrained",img)


        #print(f"{0:_^10}")
        #print(f"State: {state}")
        #print(f"Action: {action}")
        #print(f"Reward: {reward}")
        #print(f"Epoches: {epoches}")
        cv2.waitKey(1)
        time.sleep(0.2)
        epoches += 1

        if epoches == 50:
            done = True

    print(f"Steps: {epoches}")
    print(f"Penalties: {penalties}")



if __name__ == "__main__":
    envG = gym.make("Taxi-v3",render_mode="rgb_array").env   #Our Environment
    envG.reset()                                             #Reset the Environment
    cv2.imshow("Image of STATE of the Environment",envG.render())
    cv2.waitKey(0)
    #actions = env.action_space                              #Our states of actions in the Environment
    #states = env.observation_space                          #Our possibles states in the Environment
    #print(actions)
    #print(states)
    print("Running of Agent in untrained environment...")
    no_Rl(envG)                                              #Run the Agent without training (random actions)
    print("End of the time.")
    cv2.destroyAllWindows()
    print("Training of the Agent with Q-Tables...")
    Q_table = QRL(envG)
    print("Training completed!")
    print("Running of Agent...")
    run_Agent(envG,Q_table)
    print("End of the generations.")
