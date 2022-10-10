# 加载专家轨迹

# 运行专家轨迹 T0

# 运行一般轨迹 T1, T2, T3, T4, T5

# 计算所有力做的Power W0,W1,W2,W3,W4,W5,W6

# 计算所有轨迹对应的Return G1,G2,G3,G4,G5,G6

# 画图

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
from utility.cal_force import calForce1, calForce2
import cv2

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


def get_action(key):
    if key == 53:
        action = 0
    elif key == 54:
        action = 1
    elif key == 57:
        action = 2
    elif key == 56:
        action = 3
    elif key == 55:
        action = 4
    elif key == 52:
        action = 5
    elif key == 49:
        action = 6
    elif key == 50:
        action = 7
    elif key == 51:
        action = 8
    else:
        action = 0
    return action 

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
    img = cv2.imread('./data/actions.png')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Restore Model
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt_to_restore)
            print('Model Restored.')
        
        obs = env.reset() 
        # do NOT use original rewards to update policy
        for episode in range(max_episode):
            if episode % 100 == 0:
                print('Episode ', episode)
            
            observations = []
            actions = []
            rewards = []
            v_preds = []

            # 遍历这次游戏中的每一步
            obs = env.reset()
            for step in range(max_steps):
                # if episode > 1000 and episode % 100 == 0:
                #     env.render()
                env.render()
                obs = np.stack([obs]).astype(dtype=np.float32)
                # act, v_pred = Policy.get_action(obs=obs, stochastic=True)
                # act = np.asscalar(act)

                cv2.imshow('img',img)
                key = cv2.waitKey(-1)
                act = get_action(key)

                if act != 0:
                    f2_x = 0.001 * np.cos((act-1)*(np.pi/4.0))
                    f2_y = 0.001 * np.sin((act-1)*(np.pi/4.0))
                else:
                    f2_x = 0
                    f2_y = 0

                # 和环境交互
                next_obs, reward, done, info = env.step(act)

                observations.append(obs)
                actions.append(act)
                # 这里的reward并不是用来更新网络的,而是用来记录真实的
                # 表现的。
                rewards.append(reward)

                if done:
                    break
                else:
                    obs = next_obs
            v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
            # convert list to numpy array for feeding tf.placeholder

            next_observations = observations[1:]
            observations = observations[:-1]
            actions = actions[:-1]

            next_observations = np.reshape(next_observations, newshape=[-1] + list(ob_space.shape))
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            deltaS = next_observations[:,0:2] - observations[:,0:2]

            deltaF1 = calForce1(observations)
            dW1 = np.sum(deltaF1*deltaS,axis=1)
            W1 = np.sum(dW1, axis=0)

            deltaF2 = calForce2(actions)
            dW2 = np.sum(deltaF2*deltaS, axis=1)
            W2 = np.sum(dW2, axis=0)
            # print(dW2)
            W = W1+W2
            G = sum(rewards)
            print(W1,W2,W,G,1.0/G)
if __name__ == '__main__':
    main()
