import retro
import os
from ..preprocessing.stack_frame import preprocess_frame, stack_frame

class ReplayExperience:
    def __init__(self, env):
        
        #movie_path = 'human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
        movie_path = 'SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
        movie = retro.Movie(movie_path)
        self.movie = movie 
        self.movie.step()

        
        #self.env = retro.make(game=self.movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
        self.env = env
        self.env.initial_state = self.movie.get_state()
        self.env.reset()

        print('stepping movie')

    def getExperience(self, env):

        state = self.stack_frames(None, env.reset(), True)
        action = []
        reward = 0
        next_state = state
        done = False
        while self.movie.step():
        
            for i in range(len(env.buttons)):
                action.append(self.movie.get_key(i, 0))

            next_state, reward, done, info = env.step(action)
            next_state = self.stack_frames(state, next_state, False)
            #self.env.render()
            #saved_state = env.em.get_state()
        env.close()
        return state, action, reward, next_state, done

    def stack_frames(self, frames, state, is_new=False):
        frame = preprocess_frame(state, (1, -1, -1, 1), 84)
        frames = stack_frame(frames, frame, is_new)

        return frames  

