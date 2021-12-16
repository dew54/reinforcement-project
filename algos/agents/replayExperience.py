
from ..preprocessing.stack_frame import preprocess_frame, stack_frame
#import dqn_agent


class ReplayExperience:
    def __init__(self, env, movie, memory):
        
        #movie_path = 'human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
        #movie_path = 'human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
        #self.movie = retro.Movie(movie_path)
        self.movie = movie
        self.movie.step()
        self.memory = memory
        
        #self.env = retro.make(game=self.movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
        self.env = env
        self.env.initial_state = self.movie.get_state()
        self.env.reset()

        print('stepping movie')

        self.AddMemory()

    def AddMemory(self):
        state = self.stack_frames(None, self.env.reset(), True)
        while self.movie.step():
            keys = []
            
            for p in range(1):
                for i in range(self.env.num_buttons):
                    action = self.movie.get_key(i, p)
                    keys.append(action)
                    #next_state, reward, done, info = self.env.step(action)
                    
                
            next_state, reward, done, info = self.env.step(keys)
            action = self.translateAction(keys)
            print(action)
            next_state = self.stack_frames(state, next_state, False)
            self.memory.add(state, action , reward, next_state, done)
            state = next_state
            #self.env.render()







    #            #env.render()
    #            saved_state = self.env.em.get_state()
    #    self.env.close()
 
     #   self.memory.add(state, action, reward, next_state, done)
      #  return state, keys, _rew, next_state, _done, _info

    def stack_frames(self, frames, state, is_new=False):
        frame = preprocess_frame(state, (1, -1, -1, 1), 84)
        frames = stack_frame(frames, frame, is_new)

        return frames  
    def translateAction(self, action):
        
        act = self.preprocessAction(action)
        
        possible_actions = {
                    # No Operation
                    '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 0,
                    # Left
                    '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 1,
                    # Right
                    '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 2,
                    # Left, Down
                    '[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]': 3,
                    # Right, Down
                    '[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]': 4,
                    #added
                    '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 4,
                    # Down
                    '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 5,
                    # Down, B
                    '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 6,
                    # B
                    '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 7
                }

        if(possible_actions.get(act) == None):
            return 0
        else:         
            return possible_actions.get(act)    

    def preprocessAction(self, action):
        self.action = [0,0,0,0,0,0,0,0,0,0,0,0]
        for a in range(len(action)):
            if action[a] == True:
                self.action[a] = 1
            elif action[a] == False:
                self.action[a] = 0
        return str(self.action)

