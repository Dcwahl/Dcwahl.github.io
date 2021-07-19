# Connect 4

## Part 1:

So as a bit of an explanation on the background here, this initially began in an exercise in a little bit of OOP, as well as to tackle a couple of frustrations I had with a particular Kaggle lesson:

https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning

I had started the lesson just for fun, but ran into a couple of errors that I couldn't get past (if I recall correctly- it was due to package version errors, which I honestly didn't feel like troubleshooting and rectifying). My hope is to use this in the long run to learn a bit of Reinforcement Learning, or perhaps some sort of other deep learning approach that may adequately perform in the context of Connect 4.

With that said, I would suggest taking at least a quick glimpse at the above Kaggle lesson if you're going to read forward in this, as some of the choices I make in design might otherwise seem a little mysterious. 

### Some initial thoughts/design choices

The Kaggle implementation, broadly, works by taking in two 'agent' functions, which can take in attributes and the current state of the board in order to make (or not) an intelligent decision as to where to place the next piece. For example, if you wanted to make an agent that would look at all valid possible columns (i.e., columns that haven't yet been fully filled) and then randomly choose from those valid columns, an approach could look something like so:


```python
def agent_random(board):
    valid_moves = [col for col in range(len(board[0])) if board[0][col] == 0]
    return random.choice(valid_moves)
```

With that in mind, let's start playing around with the actual game object. So obviously we'll need a concept of a board (which defaults to a size of 6x7)


```python
class Connect4:
    def __init__(self, columns = 7, rows = 6, n=4):
        self.columns=columns
        self.rows=rows
        self.won = False 
        self.inarow=n
```

Additionally, I would like to be able to call a function to simulate a singular game between the two agents that I pass in.


```python
def simulate(self,agent1,agent2):
    #1. Initialization Steps
    
    #2. Main Loop
    # (check win condition at end)
    
```


```python
def simulate(self,agent1,agent2):
    #1. Initialization Steps
    self.won=False
    self.board = [[0 for i in range(self.columns)] for j in range(self.rows)]
    self.nextMove = agent1 #agent1 always starts, can potentially change this to be a random choice
    self.moveAfter = agent2
    self.piece=1 #agent1 pieces represented by 1, agent2 by 2
    #2. Main Loop
    # (check win condition at end)
```

Now, let's think through how to properly make a move. So we'll want to ask the agent for their choice (which is not guaranteed to be a random choice, notably), then check the win condition. So the main movement loop may look something like:


```python
while not self.won:
    #move
    self.make_move()
    if self.check_win():
        self.won=self.nextMove
    self.nextMove,self.moveAfter=self.moveAfter,self.nextMove
    self.piece=1 if self.piece!=1 else 2
return self.won.__name__
```


```python
def make_move(self):
    col = self.nextMove(self.board,self.piece,self.inarow) #get column choice from agent
    if not self.check_if_valid(col): #checks to see if the choice is even valid, if not that agent loses
        self.won=self.moveAfter
        return
    for i in range(self.rows-1,-1,-1): #loops from bottom up to find first available row
        if self.board[i][col]==0:
            break
    self.board[i][col] = self.piece #modify the actual board
    return
```

Now, how to figure out if win condition is satisfied? One way that quickly comes to mind is to simply look through the board a bunch seeing if you ever see the n in a row in a horizontal, diagonal, or vertical direction. This however sounds like a whole bunch of annoying looping and indices and whole bunch of stuff that I don't really want to deal with. What you could do instead is actual convolve the board with n-sized kernels representing the vertical, horizontal, and then the two diagonal flavors of win conditions. For the case of Connect 4 (as opposed to generalized Connect N), such an approach may look something like:


```python
from scipy.signal import convolve2d

def check_win(self):
    horizontal_kernel = np.array([[ 1, 1, 1, 1]])
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(4, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
    #converting all non-player positions to zero
    checkBoard=np.array(self.board)
    checkBoard[checkBoard!=self.piece]=0
    #checkBoard[checkBoard==self.piece]=1
    for kernel in detection_kernels:
        if (convolve2d(checkBoard == self.piece, kernel, mode="valid") == 4).any():
            return True
    return False
```

Quite simply, if we convolve with those filters and find that any of them result in an element equal to 4 (or more generally n for Connect N), then we know that we've hit the win condition

In all, my implementation looks something like the following;


```python
#dependencies
from scipy.signal import convolve2d
import numpy as np
import random

class Connect4:
    def __init__(self, columns = 7, rows = 6, n=4):
        self.columns=columns
        self.rows=rows
        self.won = False 
        self.inarow=n
    
    def simulateMany(self,agent1,agent2,num):
        scores={agent1.__name__:0,agent2.__name__:0}
        for i in range(num):
            winner = self.simulate(agent1,agent2)
            scores[winner]+=1
            print(winner," wins game ",i)
        print(agent1.__name__,"won: ",scores[agent1.__name__]," times")
        print(agent2.__name__,"won: ",scores[agent2.__name__]," times")
    
    def simulate(self,agent1,agent2):
        self.won=False
        self.board = [[0 for i in range(self.columns)] for j in range(self.rows)]
        self.nextMove = agent1
        self.moveAfter = agent2
        self.piece=1
        while not self.won:
            #move
            self.make_move()
            if self.check_win():
                self.won=self.nextMove
            self.nextMove,self.moveAfter=self.moveAfter,self.nextMove
            self.piece=1 if self.piece!=1 else 2
            #for i in range(self.rows):
                #print(self.board[i])
            #print('Next move: ',self.nextMove.__name__)
        return self.won.__name__
        #print(someone won!)
    
    def make_move(self):
        col = self.nextMove(self.board,self.piece,self.inarow)#(self.board,self.columns,self.rows)
        if not self.check_if_valid(col):
            self.won=self.moveAfter
            return
        #do something with col to modify self.board
        for i in range(self.rows-1,-1,-1):
            if self.board[i][col]==0:
                break
        self.board[i][col] = self.piece
        return
    
    def check_win(self):
        horizontal_kernel = np.array([[ 1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        #converting all non player positions to zero
        checkBoard=np.array(self.board)
        checkBoard[checkBoard!=self.piece]=0
        #checkBoard[checkBoard==self.piece]=1
        for kernel in detection_kernels:
            if (convolve2d(checkBoard == self.piece, kernel, mode="valid") == 4).any():
                return True
        return False
    
    def check_if_valid(self,column):
        valid = self.find_valids()
        if column in [key for key, val in valid.items() if val==True]:
            return True
        else:
            return False
    
    def find_valids(self):
        return {i:not self.board[0][i] for i in range(len(self.board[0]))}
```

There's definitely some additional changes to be made in this whole thing, but overall seems to work fine. As a proof of concept, I will now simulate 10 games between 2 random agents as described earlier. Note that despite different names they are in practice identical.


```python
def agent_random1(board,piece,inarow):
    valid_moves = [col for col in range(len(board[0])) if board[0][col] == 0]
    return random.choice(valid_moves)

def agent_random2(board,piece,inarow):
    valid_moves = [col for col in range(len(board[0])) if board[0][col] == 0]
    return random.choice(valid_moves)
```


```python
randomGame = Connect4()
```


```python
randomGame.simulateMany(agent_random1,agent_random2,num=10)
```

    agent_random2  wins game  0
    agent_random2  wins game  1
    agent_random2  wins game  2
    agent_random1  wins game  3
    agent_random2  wins game  4
    agent_random1  wins game  5
    agent_random1  wins game  6
    agent_random1  wins game  7
    agent_random1  wins game  8
    agent_random1  wins game  9
    agent_random1 won:  6  times
    agent_random2 won:  4  times


If you *really* want to convince yourself that the above is working, I would uncomment out the print statements in the simulate function to track the progress of the game.

Anyways, that's about it for this particular post, not a ton of actually interesting stuff in here thus far but I am hoping to have something interesting to show in the coming weeks.
