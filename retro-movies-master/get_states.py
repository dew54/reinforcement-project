import retro

movie_path = 'human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
movie = retro.Movie(movie_path)
movie.step()

env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
env.initial_state = movie.get_state()
env.reset()

print('stepping movie')
while movie.step():
    keys = []
    for i in range(len(env.buttons)):
        keys.append(movie.get_key(i, 0))
    _obs, _rew, _done, _info = env.step(keys)
    #env.render()
    saved_state = env.em.get_state()
    print(_obs)

print('stepping environment started at final state of movie')
env.initial_state = saved_state
env.reset()

