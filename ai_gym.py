import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from tqdm import tqdm

LR = 1e-3
env = gym.make("CartPole-v1")
env.reset()
n_pixel_slide = 500
threshold_score = 50
game_levels = 10000


def some_random_games_first():
   
    for episode in tqdm(range(5)):
        env.reset()
        
        for t in range(200):
           
            env.render()
            
            action = env.action_space.sample()
            
            observation, reward, done, info = env.step(action)
            if done:
                break
                


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in (range(game_levels)):
        score = 0
        
        game_memory = []
       
        prev_observation = []
        for _ in (range(n_pixel_slide)):
            action = random.randrange(0,2)
            
            observation, reward, done, info = env.step(action)
            
            
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

     
        if score >= threshold_score:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)
    
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    print(max(accepted_scores),' was the maximum score generated')
    return training_data
initial_population()

def neural_network_model(input_size):
    network=input_data(shape=[None,input_size,1],name='input')
    network=fully_connected(network,128,activation='relu')
    network=dropout(network,0.8)
    network=fully_connected(network,256,activation='relu')
    network=dropout(network,0.8)
    network=fully_connected(network,512,activation='relu')
    network=dropout(network,0.8)
    network=fully_connected(network,256,activation='relu')
    network=dropout(network,0.8)
    network=fully_connected(network,128,activation='relu')
    network=dropout(network,0.8)

    network=fully_connected(network,2,activation='softmax')
    network=regression(network,optimizer='adam',learning_rate=LR,
        loss='categorical_crossentropy',name='targets')
    model=tflearn.DNN(network,tensorboard_dir='log')
    return model

def train_model(training_data,model=False):
    X=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    Y=[i[1] for i in training_data]
    if not model:
        model=neural_network_model(input_size=len(X[0]))
    model.fit({'input':X} ,{'targets':Y},n_epoch=3 ,snapshot_step=500,show_metric=True)       
    return model
training_data=initial_population()
model=train_model(training_data)
scores=[]
choices=[]
for each_game in (range(10)):
    score=0
    game_memory=[]
    prev_obs=[]
    env.reset()
    for _ in (range(n_pixel_slide)):
        env.render()
        if len(prev_obs)==0:
            action=random.randrange(0,2)
        else:
            action=np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)
        new_observation,reward,done,info=env.step(action)
        prev_obs=new_observation
        game_memory.append([new_observation,action])
        score+=reward

        if done:
            break
    scores.append(score)
print('Avg Score for the games:',sum(scores)/len(scores))
print('Left movement:{}, Right movement:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
model.save('423 .npy')