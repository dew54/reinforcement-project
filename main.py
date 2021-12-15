import train

def main():

    useNEAT = False
    useDDQN = False
    useHumanExperience = False
    level = 'SonicTheHedgehog-Genesis'
    number_of_episodes = 300


    args = {
        "useNEAT": useNEAT,
        "useDDQN": useDDQN,
        "useHumanExperience": useHumanExperience,
        "level": level,
        "#episodes" : number_of_episodes
    }

    
    training = train.Train(args)
    training.train(number_of_episodes)
    
if __name__ == "__main__":
    main()