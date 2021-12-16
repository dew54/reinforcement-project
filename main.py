import train
import retro


def main():

    movie_path = 'retro-movies-master/human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
    movie = retro.Movie(movie_path)

    useNEAT = False
    useDDQN = False
    useHumanExperience = False
    level = 'GreenHillZone.Act1'
    number_of_episodes = 3
    epsilon = 0.1
    update_every = 100

    

    args = {
        "useNEAT": useNEAT,
        "useDDQN": useDDQN,
        "useHumanExperience": useHumanExperience,
        "level": level,
        "#episodes" : number_of_episodes,
        "epsilon"  : epsilon,
        "update_every" : update_every
    }

    #rom = import 'C:\Users\dew54\OneDrive\Documenti\UniTs\RL\reinforcement-project\rom'
    
    
    training = train.Train(args, movie)
    training.train(number_of_episodes)
    
if __name__ == "__main__":
    main()