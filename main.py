import train
import retro


def main():

    

    useNEAT = False
    useDDQN = False
    useHumanExperience = False
    colab = False
    level = 'GreenHillZone.Act1'
    number_of_episodes = 10
    epsilon = 0.1
    update_every = 100

    

    

    movie_path = 'retro-movies-master/human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
    if colab:
      movie_path = '/content/drive/MyDrive/Colab Notebooks/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
    
    #rom = import 'C:\Users\dew54\OneDrive\Documenti\UniTs\RL\reinforcement-project\rom'

    args = setArgs(useNEAT, useDDQN, useHumanExperience, level, number_of_episodes, epsilon, update_every, colab)
    
    print('Phase 1')
    training = train.Train(args)
    print(training.train(number_of_episodes))























    
""" 
    useNEAT = False
    useDDQN = True
    useHumanExperience = False
    print('Phase 2')
    args = setArgs(useNEAT, useDDQN, useHumanExperience, level, number_of_episodes, epsilon, update_every, colab)
    training = train.Train(args)
    training.train(number_of_episodes)

    useNEAT = False
    useDDQN = True
    useHumanExperience = True
    print('Phase 3')
    movie = retro.Movie(movie_path)
    args = setArgs(useNEAT, useDDQN, useHumanExperience, level, number_of_episodes, epsilon, update_every, colab)
    training = train.Train(args, movie)
    training.train(number_of_episodes) """


def setArgs(useNEAT, useDDQN, useHumanExperience,level, number_of_episodes, epsilon, update_every, colab):

    args = {
        "useNEAT": useNEAT,
        "useDDQN": useDDQN,
        "useHumanExperience": useHumanExperience,
        "level": level,
        "#episodes" : number_of_episodes,
        "epsilon"  : epsilon,
        "update_every" : update_every,
        "colab" : colab
    }
    return args
    
    
if __name__ == "__main__":
    main()

