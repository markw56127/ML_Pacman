#!/usr/bin/env python
# coding: utf-8

# **Hommework 26**

# In our last variation of mini-pacman the ghost will actively chase you. Pacman moves at the same speed as the ghost, so as long as he's going in the right direction he can make it to the power pellet. However, one wrong move and the ghost will get him!
# 
# We start with all the code from the previous assignment:

# In[1]:


import numpy as np


# In[2]:


class Dense():
    '''Fully connected linear layer class'''
    def __init__(self, input_size, output_size):
        #np.random.seed(input_size) #control randomness! Remove for real use
        self.kernel = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)

    def predict(self,input):
        self.input=input #need to store input for use in update
        return self.input@self.kernel+self.bias

    def backprop(self,grad_in):
        self.grad_in=grad_in #need to store this for use in update
        return grad_in@self.kernel.T

    def update(self,lr):
        #target must have shape (n,1) for regression
        kernel_grad = self.input.T @ self.grad_in
        bias_grad = np.sum(self.grad_in, axis=0)
        self.kernel -= lr * kernel_grad
        self.bias -= lr * bias_grad


# In[3]:


class ReLU():
    '''ReLU layer class'''
    def __init__(self):
        pass

    def predict(self,input):
        self.input=input
        return np.maximum(input,0)

    def backprop(self,grad_in):
        return grad_in*(self.input>0)

    def update(self,lr):
        pass


# In[4]:


class Sequential():
    def __init__(self,layerlist):
        self.layerlist=layerlist

    def add(self,layer):
        self.layerlist+=[layer]

    def predict(self,input):
        for layer in self.layerlist:
            input=layer.predict(input)
        return input

    def backprop(self,grad):
        for layer in self.layerlist[::-1]:
            grad=layer.backprop(grad)

    def update(self,lr):
        for layer in self.layerlist:
            layer.update(lr)

    def train(self,X,y,epochs,lr):
        for i in range(epochs):
            lossgrad=(self.predict(X)-y)/len(X)
            self.backprop(lossgrad)
            self.update(lr)


# In[5]:


model=Sequential([])
model.add(Dense(5,64)) #state is 4D, action is 1D, so 5D all together
model.add(ReLU())
model.add(Dense(64,32)) #You can play with the hidden states
model.add(ReLU())
model.add(Dense(32,1)) #Q_value is output


# In[6]:


def MakeNewState():
    ghost=np.random.choice(np.arange(14))+1
    ghostx=(ghost%4)+1 #random x coordinate between 1 and 4, avoiding (1,1) and (4,4)
    ghosty=(ghost//4)+1 #random y coordinate between 1 and 4, avoiding (1,1) and (4,4)
    return np.array([1,1,ghostx,ghosty]) #starting pacman position at (1,1)


# Not all starting positions of the ghost will be winnable for pacman in the latest variant of the game. When the ghost is chasing pacman, you'll have to contrain its starting position to a limited set of options. Hence, for the full version you'll want to call `MakeWinnableState` at the beginning of each episode, instead of `MakeNewState`.

# In[7]:


def MakeWinnableState():
    ghost_positions=np.array([[1,2],[2,1],[2,3],[3,2],[1,4],[4,1],[3,4],[4,3]])
    ghost=ghost_positions[np.random.choice(np.arange(8))]
    return np.array([1,1,ghost[0],ghost[1]])


# As before, each step of the game involves:
# 1. choosing an action for pacman, 
# 2. seeing what the resulting reward will be
# 3. determining the next state, and 
# 4. training the model network. 
# 
# After these steps take place, if the episode is not over then it will be time to move the ghost. Here is the function to update the ghost position so that it always moves closer to pacman. Note that in many cases there will be two viable ghost moves. This function will choose one of those two possibilities randomly, so that each time the game is played pacman will have to be "smart" enough to adapt. 

# In[8]:


def UpdateGhost(state):
    position=state[:2] #pacman position
    ghost=state[2:] #ghost position
    diff=np.sign(ghost-position)
    i=np.random.choice(np.array([0,1])[diff!=0])
    ghost[i]-=diff[i] #updates ghost to be closer to pacman


# The `NextState`, `GetReward`, and `GetQvalues` functions should be the same as in the last assignment.

# In[9]:


def NextState(state,action):
    nextstate=np.copy(state)
    if action==0: nextstate[0]-=1 #move down
    if action==1: nextstate[0]+=1 #move up
    if action==2: nextstate[1]-=1 #move left
    if action==3: nextstate[1]+=1 #move right
    return nextstate


# In[10]:


def GetReward(state,action):
    nextstate=NextState(state,action)
    if  nextstate[0]==4 and nextstate[1]==4: #powerpellet!
        return 10
    if nextstate[0]==0 or nextstate[0]==5: #top/bottom wall
        return -10
    if nextstate[1]==0 or nextstate[1]==5: #left/right wall
        return -10
    if  nextstate[0]==nextstate[2] and nextstate[1]==nextstate[3]: #ghost!!
        return -10
    else:
        return 0.1


# In[11]:


def GetQvalues(state):
    state_action=np.zeros((4,5))
    state_action[:,:4]=state #First four channels are given state
    state_action[:,4]=np.arange(4) #add all possible actions
    Q_values=model.predict(state_action).reshape((4))
    return Q_values


# Here are the parameters I recommend starting with for the Q-learning algorithm:

# In[12]:


#params
num_episodes=20000
gamma = 0.8  # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.9998


# In[32]:


state=MakeNewState()
state


# The following code is my solution to the last assignment. For this assignment, modify this code as follows:
# * for the first half of the episodes it will solve the previous problem (fixed ghost)
# * for the second half of the episodes it will continue training (fine-tuning the model) with a moving ghost. 
# 
# Places where you'll have to add some code are indicated below. 

# In[49]:


for i in range(num_episodes):
    state=MakeNewState() 
    #Add code here so that if more than half the episodes have been executed, 
    #the initial state is one that will be winnable for pacman, rather than 
    #any randomly chosen initial state.
    total_reward=0
    steps=0
    done=False
    if i > num_episodes/2:
        state = MakeWinnableState()
    
    while not done and steps<20: #start episode
        steps+=1
        
        Q_values = GetQvalues(state)
        
        if np.random.rand()<=epsilon: #Exploration
            action=np.random.choice(4) #Random action
        else: #Exploitation
            action = np.argmax(Q_values)  #Choose the best action
            
        state_action=np.append(state,action).reshape(1,-1)

        reward=GetReward(state,action)
        next_state=NextState(state,action)
        total_reward+=reward

        
        if np.abs(reward)==10:
            done=True
            newQ=reward
        else:
            newQ=reward+gamma*np.max(GetQvalues(next_state))
        
        newQ=np.array(newQ).reshape(1,-1)        
        model.train(state_action,newQ,epochs=2,lr=0.001)
   
        state=next_state
        
        #Add code here to update the ghost's position if more than half the 
        #episodes have been executed, and the current episode is not done.
        
        if i > num_episodes/2:
            state[2:] = 1,3
    
    epsilon*=epsilon_decay
    if epsilon<epsilon_min:
        epsilon=epsilon_min
        
    if i%100==0:
        print(f'episode: {i}, total reward: {total_reward}, epsilon: {epsilon:.2f}, steps: {steps}')


# Run the next two code blocks to watch pacman outsmart the ghost!

# In[50]:


from time import sleep
from IPython.display import clear_output
pac='\u15E7'
dot='\u00B7'
power='\u25EF'
ghost='\u15E3'
wall='x'

def MakeBoard(state):
    board=[]
    for i in range(6):
        board.append([])
        for j in range(6):
            board[i].append(dot)
    for i in range(6):
        board[0][i]=wall
        board[5][i]=wall
        board[i][0]=wall
        board[i][5]=wall
            
    board[state[2]][state[3]]=ghost
    board[4][4]=power
    board[state[0]][state[1]]=pac

    out=''
    for i in range(6):
        out+=''.join(board[i])+'\n'
    return out

def ComputerPlayPacMan():
    state=MakeWinnableState()

    print(MakeBoard(state))
    steps=0
    while True and steps<20:
        steps+=1
        action = np.argmax(GetQvalues(state))
        reward = GetReward(state,action)
        state = NextState(state,action)
        sleep(1)
        clear_output(wait=True)
        print(MakeBoard(state))
        if np.abs(reward)==10:
            break
        UpdateGhost(state)
        sleep(1)
        clear_output(wait=True)
        print(MakeBoard(state))


# In[51]:


ComputerPlayPacMan()

