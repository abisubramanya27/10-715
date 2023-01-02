import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys

MAX_ITERATIONS = 500
ENV = "FrozenLake-v1"


def qlearning(env=ENV, epsilon=0.7, alpha=0.8, gamma=0.9):
    env = gym.make(env, is_slippery=False, render_mode="rgb_array")

    na = env.action_space.n
    ns = env.observation_space.n
    Q = np.zeros((ns, na))
    random.seed(42)
    for i in range(MAX_ITERATIONS):
        # TODO: write your code here to update Q
        cur_state, _ = env.reset()
        term = False
        while not term:
            # eps-greedy sampling of action
            best_action = np.argmax(Q[cur_state])
            if random.random() <= epsilon: # Exploration
                cur_act = random.randint(0, na-1)
            else: # Exploitation
                cur_act = best_action
            next_state, rew, term, _, _ = env.step(cur_act)

            # Update step
            Q[cur_state, cur_act] = (1 - alpha)*Q[cur_state, cur_act] + alpha*(rew + gamma*np.max(Q[next_state]))
            cur_state = next_state

    env.close()
    return Q

def plot(state_image, idx, epsilon):
    plt.figure()
    plt.imshow(state_image)
    plt.axis('off')
    plt.title(rf'$\epsilon$: {epsilon} | State {idx}')
    if not os.path.exists(f'./ql_plots/eps_{epsilon}'):
        os.makedirs(f'./ql_plots/eps_{epsilon}')
    plt.savefig(f'./ql_plots/eps_{epsilon}/state_{idx}.png')
    plt.close()

def generate_trajectory(Q, epsilon=0.7, env=ENV, gamma=0.9, seed=2022):
    env = gym.make(env, is_slippery=False, render_mode="rgb_array")
    cur_state, _ = env.reset()
    na = env.action_space.n
    idx = 0
    plot(env.render(), idx, epsilon)
    term, tot_rewards, discount = False, 0, 1
    random.seed(seed)
    while not term and idx < 10:
        # idx < 10 to prevent agent stuck scenario to run indefinitely (when eps = 0 mainly)
        idx += 1
        best_action = np.argmax(Q[cur_state])
        if random.random() <= epsilon:  # Exploration
            cur_act = random.randint(0, na - 1)
        else:  # Exploitation
            cur_act = best_action

        next_state, rew, term, _, _ = env.step(cur_act)
        plot(env.render(), idx, epsilon)
        print(cur_state, cur_act, next_state, rew, term)

        cur_state = next_state
        tot_rewards += discount*rew
        discount *= gamma

    env.close()
    print(f'Total reward for the trajectory: {tot_rewards}')

if __name__ == '__main__':
    # Train Q with Q-learning
    Q = qlearning()
    # TODO: write your code here to evaluate and render environment
    for epsilon in [0, 0.25, 0.5]:
        generate_trajectory(Q, epsilon=epsilon, seed=67)
