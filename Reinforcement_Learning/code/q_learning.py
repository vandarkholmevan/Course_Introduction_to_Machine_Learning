from environment import MountainCar
import numpy as np
import copy
import sys

MODE       = sys.argv[1]
WEIGHT_OUT = sys.argv[2]
RETURN_OUT = sys.argv[3]
EPISODES   = int(sys.argv[4])
MAX_ITERATIONS = int(sys.argv[5])
EPSILON    = float(sys.argv[6])
GAMMA      = float(sys.argv[7])
LRATE      = float(sys.argv[8])

class Player():
    def __init__(self, env, mode, episodes, max_iter, epsilon, gamma, lrate):
        self.env = MountainCar(mode)
        self.mode = mode
        self.episodes = episodes
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gamma = gamma
        self.lrate = lrate
        if self.mode == "raw":
            self.weight = np.zeros((2, 3))
        elif self.mode == "tile":
            self.weight = np.zeros((2048, 3))    
        self.bias = 0
        self.actBest = 0
        self.actReal = 0
        self.returns = ({}, 0, 0)
        self.nextReward = 0
        self.returnsTemp = []
        self.returnsFinal = []
    
    def calQ(self, actionIdx):
        Q = self.bias
        for key, value in self.nextState.items():
            Q += self.weight[key, actionIdx] * value
        return Q
        
    def findActBest(self):
        self.Q_li = [self.calQ(0), self.calQ(1), self.calQ(2)]
        self.actBest = self.Q_li.index(max(self.Q_li))
           
    def actSelection(self):
        p_best = 1 - self.epsilon
        p_temp = self.epsilon / 3
        self.actReal = np.random.choice([self.actBest, 0, 1, 2], 1, p=[p_best, p_temp, p_temp, p_temp])[0]
        self.Q = self.Q_li[self.actReal]
        
    def actExe(self):
        self.states = copy.deepcopy(self.nextState)
        self.returns = self.env.step(self.actReal)
        self.nextState = copy.deepcopy(self.returns[0])
        self.nextReward = self.returns[1]
        self.returnsTemp.append(self.nextReward)
        
    def calTD(self):
        self.nextQ = max([self.calQ(0), self.calQ(1), self.calQ(2)])
        self.TD = self.Q - (self.nextReward + self.gamma * self.nextQ)
        
    def uptWeight(self):
        self.calTD()
        for key, value in self.states.items():         
            self.weight[key, self.actReal] -= self.lrate * self.TD * value
        self.bias -= self.lrate * self.TD
    
    def runOneEpsd(self):
        self.nextState = copy.deepcopy(self.env.reset())
        self.states = {}
        self.returnsTemp = []
        self.returns = ({}, 0, 0)
        step = 0
        while ((step < self.max_iter) and (self.returns[-1] == 0)):
            
            self.findActBest()
            self.actSelection()
            self.actExe()
            self.uptWeight()
            step += 1
        return sum(self.returnsTemp)
          
    def train(self):
        for i in range(self.episodes):
            self.returnsFinal.append(self.runOneEpsd()) 
    
    def writeWeight(self, filename):
        f = open(filename, 'w')
        f.write(str(self.bias) + '\n')
        for row in self.weight:
            for w in row:
                f.write(str(w) + '\n')
        f.close() 
    def writeReward(self, filename):
        f = open(filename, 'w')
        for rwd in self.returnsFinal:
            f.write(str(rwd) + '\n')
        f.close()

def main():
    player = Player(MountainCar, MODE, EPISODES, MAX_ITERATIONS, EPSILON, GAMMA, LRATE)
    player.train()
    player.writeWeight(WEIGHT_OUT)
    player.writeReward(RETURN_OUT)

if __name__ == "__main__":
    main()