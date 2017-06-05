#!/usr/bin/env python

import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import random
import numpy as np
from collections import deque
import os
from PIL import Image
import csv
from os.path import join,getsize,getmtime
import time
import scipy.io as scio
import matplotlib.pyplot as plt

GAME = 'tetris' # the name of the game being played for log files
ACTION_SIZE = 3 # number of valid actions
GAMMA = 0.85 # decay rate of past observations
OBSERVE = 4000 # timesteps to observe before training
EXPLORE = 4000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 0.05 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
ACTIONS = np.array([1,0,-1])  #long,neutral,short
TRAIN_IMG_PATH = "./dataset/AMZN/AMZN_TRAIN/AMZN_PIC"
TRAIN_MAT_PATH = "./dataset/AMZN/AMZN_TRAIN/AMZN.mat"
TRAIN_REWARD_PATH = "./dataset/AMZN/AMZN_TRAIN/earning.csv"
TEST_IMG_PATH = "./dataset/AMZN/AMZN_TEST/AMZN_PIC"
TEST_MAT_PATH = "./dataset/AMZN/AMZN_TEST/AMZN.mat"
TEST_REWARD_PATH = "./dataset/AMZN/AMZN_TEST/earning.csv"
TEST = True
TRAIN = False

def imgs2tensor(root_dir):
    states = []
    file_list = os.listdir(root_dir)
    path_dict = {}
    for i in range(len(file_list)):
        path_dict[file_list[i]] = getmtime(join(root_dir,file_list[i]))
    sort_list = sorted(path_dict.items(),key=lambda e:e[1],reverse=False)
    for i in range(0,len(file_list)):
        path = os.path.join(root_dir,sort_list[i][0])
        print("path is :",path)
        if os.path.isfile(path):
            img = cv2.imread("{}".format(path))
            img_gray = cv2.cvtColor(cv2.resize(img,(80,80)),cv2.COLOR_BGR2GRAY)
            ret, data = cv2.threshold(img_gray,1,255,cv2.THRESH_BINARY)
            states.append(data)
    print("states shape:",np.shape(states))
#           data = cv2.cvtColor(cv2.resize(data_new, (80, 80)), cv2.COLOR_BGR2GRAY)
    states_size = np.shape(states)[0]
    return states

def getFileOrderByUpdate(path):
    file_list = os.listdir(path)
    path_dict = {}
    for i in range(len(file_list)):
        path_dict[file_list[i]] = getmtime(join(path,file_list[i])) 
    sort_list = sorted(path_dict.items(),key=lambda e:e[1],reverse=False)
    for i in range(len(sort_list)):
        print(sort_list[i][0],sort_list[i][1])

def get_reward(file_dir):
    reward = []
    with open(file_dir,'r') as file:
        reader = csv.reader(file)
        for line in reader:
            reward.append(line[0])
    reward_float = [float(str) for str in reward]
    return reward_float

def create_csv_file(file_name="",data_list=[]):
    with open(file_name,"w") as csv_file:
        csv_writer = csv.writer(csv_file)
#        for data in data_list:
#            print("data is:",data)
        csv_writer.writerows(map(lambda x:[x],data_list))
        csv_file.close

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTION_SIZE])
    b_fc2 = bias_variable([ACTION_SIZE])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 1])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)


    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    #define the cost function
    a = tf.placeholder("float", [None, ACTION_SIZE])
    y = tf.placeholder("float", [None])
    #readout_action = tf.reduce_sum(tf.matmul(readout, a), reduction_indices = 1)
    readout_action = tf.reduce_sum(tf.matmul(readout,a,transpose_b=True), reduction_indices = 1) 
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # store the previous observations in replay memory
    D = deque()
#    if TRAIN:
#       if os.path.exists(TRAIN_MAT_PATH):
#            states = scio.loadmat(TRAIN_MAT_PATH)
#       else:
    states = imgs2tensor(TEST_IMG_PATH)
#            states_dict = {"states":states}
#           scio.savemat("./dataset/AMZN/AMZN_TRAIN/AMZN.mat",states_dict)
#    if TEST:
#        if os.path.exists(TEST_MAT_PATH):
#            states = scio.loadmat(TEST_MAT_PATH)
#        else:
#            states = imgs2tensor(TEST_IMG_PATH)
#            scio.savemat("./dataset/AMZN/AMZN_TEST/AMZN.mat",states)

    states_size = np.shape(states)[0]     
    reward = get_reward(TRAIN_REWARD_PATH)
   
    s_t = np.reshape(states[0],(80,80,1))
    
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks_1")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = 0
    cum_reward = []
    
    while "pigs" != "fly":
        # choose an action epsilon greedily
        if t<states_size-1:
            terminal = True
            readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
            a_t = np.zeros([ACTION_SIZE])
            action_index = 0
            if random.random() <= epsilon or t <= OBSERVE:
                action_index = random.randrange(ACTION_SIZE)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            for i in range(0, K):
                # run the selected action and observe next state and reward
                s_t1 = np.reshape(states[t+1],(80,80,1))
                r_t = reward[t]*(np.dot(a_t,ACTIONS.T))
                
                # store the transition in D
                D.append((s_t, a_t, r_t, s_t1))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()

        # only train if done observing
        if t>BATCH:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
            # if terminal only equals reward
            # if minibatch[i][4]:
            #     y_batch.append(r_batch[i])
            # else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})


        # save progress every 10000 iterations
        if TRAIN==True:
            if t % 50 == 0:
                saver.save(sess, 'saved_networks_1/' + '{}-dqn'.format(time.strftime('%d-%h-%m',time.localtime(time.time()))),global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
        if TEST==True:
            if t>0:
                cum_reward.append(cum_reward[t-1]+r_t)
            else :
                cum_reward.append(r_t)
            if t==states_size-1:
                count = 0
                for i in cum_reward:
                    if i>0:
                        count+=1
                print("accuracy rate:",count/states_size)
                t_label = np.linspace(0,1,states_size)
#               plt.plot(t_label,cum_reward,color='red',linewidth=2)
                plt.legend() 
                plt.plot(cum_reward)
                plt.xlabel("Date")
                plt.ylabel("Total Return")
                plt.title("AMAZON")
                create_csv_file("./total_return.csv",cum_reward)
                plt.show()

        s_t = s_t1
        t += 1
        
def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
#    imgs2tensor("./dataset/test")
