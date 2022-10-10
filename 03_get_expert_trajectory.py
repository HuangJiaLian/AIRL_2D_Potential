'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-15 14:31:53
@LastEditors: Jack Huang
@LastEditTime: 2019-11-22 17:08:05
'''
from irlenv import irlEnv 
import cv2
import numpy as np 
import os 


def open_file_and_save(file_path, data):
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')

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

def cal_forces(state, action_num, unitforce):
    [x,y,v_x,v_y] = state
    # Minus Partial V Partial x & Minus Partial V Partial y
    MPVPx = 4*x*(1-x**2-y**2) + (2.0*x*(y**2))/((x**2 + y**2)**2)
    MPVPy = 4*y*(1-x**2-y**2) + (2.0*(y**3))/((x**2 + y**2)**2) -(2.0*y)/(x**2 + y**2)
    f1_x = -1*MPVPx
    f1_y = -1*MPVPy
    if action_num != 0:
        f2_x = unitforce * np.cos((action_num-1)*(np.pi/4.0))
        f2_y = unitforce * np.sin((action_num-1)*(np.pi/4.0))
    else:
        f2_x = 0
        f2_y = 0       
    return f1_x, f1_y, f2_x, f2_y

def main():
    exp_traj_path = 'exp_traj'
    if os.path.exists(exp_traj_path) != True:
        os.mkdir(exp_traj_path)
    
    env = irlEnv.IRLEnv()
    observation_space = env.observation_space
    action_space = env.action_space
    print(observation_space, action_space)

    # episodes = 20
    episodes = 2

    max_steps = 500
    unitforce = 0.001
    scorces = []
    render = True 
    img = cv2.imread('./imgs/actions.png')

    for episode in range(episodes):
        state = env.reset()
        score = 0 
        done = False 

        observations = []
        actions = []
        returns = []
        action_nums = []
        print('#######################')
        print('## Episode ', episode)
        print('#######################')
        for step in range(max_steps):
            env.render()
            cv2.imshow('img',img)
            key = cv2.waitKey(-1)
            action_num = get_action(key)
            print("##########################################################")
            print("# state: ", state)
            print("# action_num: ", action_num)
            f1_x, f1_y, f2_x, f2_y = cal_forces(state, action_num, unitforce)
            print("# actions: ", f1_x, f1_y, f2_x, f2_y)
            observations.append(state)
            actions.append([f1_x, f1_y, f2_x, f2_y])
            action_nums.append(action_num)
            # Interface to the environment
            next_state, reward, done, _= env.step(action_num)
            score += reward
            state = next_state
            if done:
                break
        print('Return:', score)
        scorces.append(score)

        # All for s'
        next_observations = observations[1:]
        observations = observations[:-1]
        actions = actions[:-1]
        action_nums = action_nums[:-1]

        next_observations = np.reshape(next_observations, newshape=[-1] + list(observation_space.shape))
        observations = np.reshape(observations, newshape=[-1] + list(observation_space.shape))
        actions = np.reshape(actions, newshape=[-1] + [4])
        action_nums = np.array(action_nums).astype(dtype=np.int32)

        open_file_and_save('{}/next_observations_ep_'.format(exp_traj_path) + str(episode) + '.csv', next_observations)
        open_file_and_save('{}/observations_ep_'.format(exp_traj_path) + str(episode) + '.csv', observations)
        open_file_and_save('{}/actions_ep_'.format(exp_traj_path) + str(episode) + '.csv', actions)
        open_file_and_save('{}/action_nums_ep_'.format(exp_traj_path) + str(episode) + '.csv', action_nums)


if __name__ == "__main__":
    main()