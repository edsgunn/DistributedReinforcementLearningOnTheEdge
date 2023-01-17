import gymnasium as gym
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    print(f"Actions: {env.action_space}")
    action = int(input("Choose action: "))
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    if terminated or truncated:
        observation, info = env.reset()
env.close()

