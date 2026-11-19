'''
script for training the agent for snake using various methods
'''
# run on cpu
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
#import tensorflow as tf
import torch
from agent import DeepQLearningAgent, PolicyGradientAgent,\
                AdvantageActorCriticAgent, mean_huber_loss
import json

# some global variables
#tf.random.set_seed(42)
np.random.seed(42)
torch.manual_seed(42)
# Short integration test, enabled from the command line
SMOKE_TEST = os.environ.get("GA02_SMOKE", "0") == "1"
BENCH_TEST = os.environ.get("GA02_BENCH", "0") == "1"
CONTINUE_TEST = os.environ.get("GA02_CONTINUE", "0") == "1"
T12_TEST = os.environ.get("GA02_T12", "0") == "1"

RESUME_ITERATION = 200_000

assert sum([
    SMOKE_TEST,
    BENCH_TEST,
    CONTINUE_TEST,
    T12_TEST,
]) <= 1, "Only one experiment mode can be active."

assert not (SMOKE_TEST and BENCH_TEST), \
    "Only one test mode can be active."
version = 'v17.1'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# define no of episodes, logging frequency
#episodes = 10 if SMOKE_TEST else 2 * (10**5)
#log_frequency = 5 if SMOKE_TEST else 500
episodes = (
    10 if SMOKE_TEST
    else 500 if BENCH_TEST
    else 50_000 if CONTINUE_TEST
    else 400_000 if T12_TEST
    else 200_000
)

log_frequency = (
    5 if SMOKE_TEST
    else 250 if BENCH_TEST
    else 500
)
games_eval = 8
#episodes = 2 * (10**5)
#log_frequency = 500
#games_eval = 8

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                           buffer_size=buffer_size, version=version)

# Set output names for smoke tests and full training
'''run_tag = (
    f"{version}_smoke_{agent._device.type}"
    if SMOKE_TEST else version
)'''
if SMOKE_TEST:
    run_tag = f"{version}_smoke_{agent._device.type}"
elif BENCH_TEST:
    run_tag = f"{version}_bench_{agent._device.type}"
elif CONTINUE_TEST:
    run_tag = f"{version}_continue_{agent._device.type}"
elif T12_TEST:
    run_tag = f"{version}_t12_fresh_{agent._device.type}"
else:
    run_tag = f"{version}_train_{agent._device.type}"

os.makedirs("model_logs", exist_ok=True)

mode = (
    "SMOKE" if SMOKE_TEST
    else "BENCH" if BENCH_TEST
    else "CONTINUE" if CONTINUE_TEST
    else "T12" if T12_TEST
    else "FULL"
)

print(f"Mode: {mode}, device: {agent._device}, iterations: {episodes}")
#print(
#    f"Mode: {'SMOKE' if SMOKE_TEST else 'FULL'}, "
#    f"device: {agent._device}, iterations: {episodes}")


# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
        # buffer_size=2000, version=version)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                                  # buffer_size=10000, version=version)
# agent.print_models()

# check in the same order as class hierarchy
if(isinstance(agent, DeepQLearningAgent)):
    agent_type = 'DeepQLearningAgent'
if(isinstance(agent, PolicyGradientAgent)):
    agent_type = 'PolicyGradientAgent'
if(isinstance(agent, AdvantageActorCriticAgent)):
    agent_type = 'AdvantageActorCriticAgent'
print('Agent is {:s}'.format(agent_type))

# setup the epsilon range and decay rate for epsilon
# define rewrad type and update frequency, see utils for more details
if(agent_type in ['DeepQLearningAgent']):
    epsilon, epsilon_end = 1, 0.01
    reward_type = 'current'
    sample_actions = False
    #n_games_training = 8*16
    n_games_training = 16 if SMOKE_TEST else 8*16
    decay = 0.97
    if(supervised):
        # lower the epsilon since some starting policy has already been trained
        epsilon = 0.01
        # load the existing model from a supervised method
        # or some other pretrained model
        agent.load_model(file_path='models/{:s}'.format(version))
        # agent.set_weights_trainable()
if(agent_type in ['PolicyGradientAgent']):
    epsilon, epsilon_end = -1, -1
    reward_type = 'discounted_future'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 16
    decay = 1
if(agent_type in ['AdvantageActorCriticAgent']):
    epsilon, epsilon_end = -1, -1
    reward_type = 'current'
    sample_actions = True
    exploration_threshold = 0.1
    n_games_training = 32
    decay = 1

# GA02-T11: Continue training from the existing checkpoint
if CONTINUE_TEST:
    assert agent._device.type == "cpu", (
        "T11 is configured to resume the CPU checkpoint."
    )

    agent.load_model(
        file_path="models/v17.1_train_cpu",
        iteration=RESUME_ITERATION,
    )

    # Restore the exploration rate reached at the end of T08.
    epsilon = epsilon_end

    print(f"Loaded checkpoint at iteration {RESUME_ITERATION}")
    print(f"Resume epsilon: {epsilon}")

# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)
run_start = time.perf_counter()
# use only for DeepQLearningAgent
if(agent_type in ['DeepQLearningAgent']):
    # play some games initially to fill the buffer
    # or load from an existing buffer (supervised)
    if(supervised):
        try:
            agent.load_buffer(file_path='models/{:s}'.format(version), iteration=1)
        except FileNotFoundError:
            pass
    else:
        # setup the environment
        #games = 512
        games = 16 if SMOKE_TEST else 512
        env = SnakeNumpy(board_size=board_size, frames=frames, 
                    max_time_limit=max_time_limit, games=games,
                    frame_mode=True, obstacles=obstacles, version=version)
        ct = time.time()
        _ = play_game2(env, agent, n_actions, n_games=games, record=True,
                       epsilon=epsilon, verbose=True, reset_seed=False,
                       frame_mode=True, total_frames=games * (8 if SMOKE_TEST else 64)) # total_frames=games*64 -> original
        print('Playing {:d} frames took {:.2f}s'.format(games*64, time.time()-ct))

env = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=n_games_training,
            frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=games_eval,
            frame_mode=True, obstacles=obstacles, version=version)

# training loop
model_logs = {'iteration':[], 'reward_mean':[],
              'length_mean':[], 'games':[], 'loss':[]}
for index in tqdm(range(episodes)):

    global_iteration = (
    RESUME_ITERATION + index + 1
    if CONTINUE_TEST
    else index + 1
    )

    if(agent_type in ['DeepQLearningAgent']):
        # make small changes to the buffer and slowly train
        _, _, _ = play_game2(env, agent, n_actions, epsilon=epsilon,
                       n_games=n_games_training, record=True,
                       sample_actions=sample_actions, reward_type=reward_type,
                       frame_mode=True, total_frames=n_games_training, 
                       stateful=True)
        loss = agent.train_agent(batch_size=64,
                                 num_games=n_games_training, reward_clip=True)

    if(agent_type in ['AdvantageActorCriticAgent']):
        # play a couple of games and train on all
        _, _, total_games = play_game2(env, agent, n_actions, epsilon=epsilon,
                       n_games=n_games_training, record=True,
                       sample_actions=sample_actions, reward_type=reward_type,
                       frame_mode=True, total_games=n_games_training*2)
        loss = agent.train_agent(batch_size=agent.get_buffer_size(), 
                                 num_games=total_games, reward_clip=True)

    if(agent_type in ['PolicyGradientAgent', 'AdvantageActorCriticAgent']):
        # for policy gradient algorithm, we only take current episodes for training
        agent.reset_buffer()

    # check performance every once in a while
    if((index+1)%log_frequency == 0):
        # keep track of agent rewards_history
        current_rewards, current_lengths, current_games = \
                    play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                               record=False, sample_actions=False, frame_mode=True, 
                               total_frames=-1, total_games=games_eval)
        
        model_logs['iteration'].append(global_iteration)
        model_logs['reward_mean'].append(round(int(current_rewards)/current_games, 2))
        # model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        model_logs['length_mean'].append(round(int(current_lengths)/current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']]\
          .to_csv(f"model_logs/{run_tag}.csv", index=False)
          #.to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        #agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        agent.save_model(file_path=f"models/{run_tag}", iteration=global_iteration)
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)

# End of training benchmark
elapsed = time.perf_counter() - run_start

print(f"Total elapsed time: {elapsed:.2f} seconds")
print(f"Overall throughput: {episodes / elapsed:.2f} iterations/s")