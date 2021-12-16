import train

def main():

    useNEAT = False
    useDDQN = True
    useHumanExperience = True
    level = 'GreenHillZone.Act1'
    number_of_episodes = 200
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

    
    training = train.Train(args)
    training.train(number_of_episodes)
    
if __name__ == "__main__":
    main()