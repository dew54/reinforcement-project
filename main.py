import train
import retro


def main():

    

    useNEAT = False
    useDDQN = True
    useHumanExperience = False
    colab = True
    level = 'GreenHillZone.Act1'
    number_of_episodes = 2
    epsilon = 0.1
    update_every = 100

    

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

    movie_path = 'retro-movies-master/human/SonicTheHedgehog-Genesis/contest/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
    if colab:
      movie_path = '/content/drive/MyDrive/Colab Notebooks/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'
    movie = retro.Movie(movie_path)
    #rom = import 'C:\Users\dew54\OneDrive\Documenti\UniTs\RL\reinforcement-project\rom'
    
    
    training = train.Train(args, movie)
    training.train(number_of_episodes)
    
if __name__ == "__main__":
    main()