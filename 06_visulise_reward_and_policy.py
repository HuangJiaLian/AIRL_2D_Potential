'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 19:27:08
@LastEditors: Jack Huang
@LastEditTime: 2019-11-18 19:24:52
'''

import tensorflow as tf 
import numpy as np 
import os
import algo.generator as gen 
import algo.discriminator as dis 
import utility.logger as log 
import matplotlib.pyplot as plt 
from irlenv import irlEnv 

def get_probabilities(policy, observations, actions):
    # Evaluate distribution
    distributions = policy.get_distribution(observations)
    # Fancy Index to get probabilities
    probabilities = distributions[np.arange(distributions.shape[0]), actions]
    return probabilities

def sample_batch(*args, batch_size=32):
    N = args[0].shape[0]
    batch_idxs = np.random.randint(0, N, batch_size)
    return [data[batch_idxs] for data in args]

def drawRewards(D, episode, path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path) 
    x_min =  -1.2
    x_max = 1.2
    y_min = -0.2 
    y_max = 1.2

    n = 100

    x = np.linspace(x_min,x_max,n)
    y = np.linspace(y_min,y_max,n)

    X,Y = np.meshgrid( x , y )

    X_ = np.reshape(X,(-1,1))
    Y_ = np.reshape(Y,(-1,1))
    if D.caseNum == 0:
        obs = np.concatenate((X_,Y_),axis=1) 
    else:
        # Todo 
        pass 

    plt.clf()
    plt.title('reward function')
    plt.xlabel('x')
    plt.ylabel('y')
    # Draw reward function 
    # Prepare x positions 
    # Get rewards 
    R_ = D.get_scores(obs_t=obs)
    R = np.reshape(R_,(n,n))
    # Plot 
    cmap=plt.cm.get_cmap('inferno') 
    plt.pcolormesh(X, Y, R, cmap = cmap)
    plt.colorbar()
    plt.savefig(os.path.join(path, str(episode) + '_learned_rewards.png'))
    plt.clf()


# Load Experts Demonstration
def load_experts(path = 'exp_traj', expNum = 20):
    fileName1 = os.path.join(path, 'observations_ep_0.csv')
    fileName2 = os.path.join(path, 'next_observations_ep_0.csv')
    fileName3 = os.path.join(path, 'action_nums_ep_0.csv')
    expert_observations = np.genfromtxt(fileName1)
    next_expert_observations = np.genfromtxt(fileName2)
    expert_actions = np.genfromtxt(fileName3)

    # Concatenate
    for num in range(1,expNum):
        fileName1 = os.path.join(path, 'observations_ep_'+ str(num) +'.csv')
        fileName2 = os.path.join(path, 'next_observations_ep_'+ str(num) +'.csv')
        fileName3 = os.path.join(path, 'action_nums_ep_'+ str(num) +'.csv')
        temp1 = np.genfromtxt(fileName1)
        temp2 = np.genfromtxt(fileName2)
        temp3 = np.genfromtxt(fileName3)
        expert_observations = np.concatenate((expert_observations,temp1),axis=0)
        next_expert_observations = np.concatenate((next_expert_observations,temp2),axis=0)
        expert_actions = np.concatenate((expert_actions,temp3),axis=0)
    expert_actions = np.array(expert_actions).astype(dtype=np.int32)
    # Return expert data
    return expert_observations, next_expert_observations, expert_actions

def main():
    # Env 
    env = irlEnv.IRLEnv()
    ob_space = env.observation_space
    
    # For Reinforcement Learning
    Policy = gen.Policy_net('policy', env)
    Old_Policy = gen.Policy_net('old_policy', env)
    PPO = gen.PPO(Policy, Old_Policy, gamma=0.95)
    
    # For Inverse Reinforcement Learning
    D = dis.Discriminator(env)
    expert_observations, next_expert_observations, expert_actions = load_experts()
    
    # No use
    mean_expert_return = -300
    
    max_episode = 500
    max_steps = 500
    saveReturnEvery = 100
    saveTraceEvery = 100
    num_expert_tra = 20 


    # Saver to save all the variables
    model_load_path = './model/'
    model_name = '2d-potential'
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_load_path)
    if ckpt and ckpt.model_checkpoint_path:
            print('Found Saved Model.')
            # 669 674 1034 1037 1039
            ckpt_to_restore = ckpt.all_model_checkpoint_paths[669]
    else:
        print('No Saved Model. Exiting')
        exit()

    # Logger 用来记录训练过程
    logger_path = './trainingLog/'
    train_logger = log.logger(logger_name='2D_PSur_Testing_Log', 
        logger_path= logger_path, col_names=['Episode', 'Actor(D)', 'Expert Mean(D)','Actor','Expert Mean'])
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Restore Model
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt_to_restore)
            print('Model Restored.')
        drawRewards(D, 1, './learned_reward/')


if __name__ == '__main__':
    main()
