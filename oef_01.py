import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gym

from Q_learning_agent import QLearningAgent

# Omgeving aanmaken
env_not_human = gym.make('CartPole-v0')
env_human = gym.make('CartPole-v0', render_mode = "human")


alpha: float = 0.1
gamma: float = 0.9
decay: float = 0.9999
nr_states: int = 10**4
exploration_rate: float = 0.999


# Agent aanmaken
agent: QLearningAgent = QLearningAgent(nr_states, alpha, gamma, exploration_rate, decay)
possible_states = [[-2.4, 2.4], [-4,4], [-0.4181, 0.4181], [-4, 4]]
agent.create_bins(possible_states, 10)


aantal_episodes: int = 1000
aantal_stappen_per_episode: list = []
max_aantal_stappen_per_episode: int = 200

for episode in range(aantal_episodes):
    agent.change_exploration()
    aantal_stappen: int = 0

    # Omgeving resetten
    env = env_human if episode % 100 == 0 else env_not_human
    observatie, _ = env.reset()
    done = False

    # Iteratie
    while not done:
        aantal_stappen += 1
        # Actie kiezen
        actie = agent.compute_action(observatie)

        # Actie uitvoeren
        observatie_nu, reward, terminated, truncated, _ = env.step(actie)

        if terminated or truncated:
            done = True
            aantal_stappen_per_episode.append(aantal_stappen)

        # Reward aanpassen
        if aantal_stappen < max_aantal_stappen_per_episode and done is True:
            reward = -100

        # Agent trainen
        agent.train_agent(observatie, observatie_nu, reward)
        observatie = observatie_nu

        

max_aantal_stappen: int = max(aantal_stappen_per_episode)
min_aantal_stappen: int = min(aantal_stappen_per_episode)
gemiddeld_aantal_stappen = sum(aantal_stappen_per_episode)/len(aantal_stappen_per_episode)

print(f"max aantal stappen: {max_aantal_stappen}")
print(f"min aantal stappen: {min_aantal_stappen}")
print(f"gemiddeld aantal stappen: {gemiddeld_aantal_stappen}")

test = pd.DataFrame(data={"aantal_episodes": range(aantal_episodes), "aantal_stappen_per_episode": aantal_stappen_per_episode})

plt.bar(range(aantal_episodes), aantal_stappen_per_episode, label="Aantal stappen", width=1)
plt.title("Aantal stappen per episode")
plt.xlabel("Aantal episodes")
plt.ylabel("Aantal stappen")
plt.show()

sns.boxplot(data=aantal_stappen_per_episode)
plt.show()


env_not_human.close()
env_human.close()