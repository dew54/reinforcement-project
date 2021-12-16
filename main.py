import train

def main():

    useNEAT = False
    useDDQN = False
    useHumanExperience = True
    level = 'SonicTheHedgehog-Genesis'
    number_of_episodes = 500
    epsilon = 0.1


    args = {
        "useNEAT": useNEAT,
        "useDDQN": useDDQN,
        "useHumanExperience": useHumanExperience,
        "level": level,
        "#episodes" : number_of_episodes,
        "epsilon"  : epsilon
    }

    
    training = train.Train(args)
    training.train(number_of_episodes)
    
if __name__ == "__main__":
    main()