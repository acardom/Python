import random

class DonkeyKongInverso:
    def __init__(self):
        self.rows = 6
        self.cols = 6
        self.start = (0, 0)
        self.goal = (5, 5)
        
        # Corrección: Las escaleras deben abarcar desde la fila 0 hasta la fila 5 
        # para conectar el inicio (0,0) con el suelo de la meta (5,5)
        self.ladders = {
            (0, 2): (5, 2), (5, 2): (0, 2),
            (0, 4): (5, 4), (5, 4): (0, 4)
        }
        
        self.holes = [(2, 1), (2, 3)]
        self.state = None
        self.stochastic_prob = 0.0

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if random.random() < self.stochastic_prob:
            return self.state, -1, False

        r, c = self.state
        
        # 0=arriba, 1=abajo, 2=izquierda, 3=derecha
        if action == 2:    # Izquierda
            c = max(c - 1, 0)
        elif action == 3:  # Derecha
            c = min(c + 1, self.cols - 1)
        elif action == 0:  # Arriba
            if (r, c) in self.ladders:
                r, c = self.ladders[(r, c)]
        elif action == 1:  # Abajo
            if (r, c) in self.ladders:
                r, c = self.ladders[(r, c)]

        new_state = (r, c)
        
        if new_state == self.goal:
            reward = 20
            done = True
        elif new_state in self.holes:
            reward = -1
            done = True
        else:
            reward = -1
            done = False
            
        self.state = new_state
        return new_state, reward, done

    def state_to_idx(self, state):
        return state[0] * self.cols + state[1]