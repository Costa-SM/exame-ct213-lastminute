%% Get data
data = readtable("breakoutlog/progress.csv");

%% Separate data
rewards = data(:, "rollout_ep_rew_mean");
length = data(:, "rollout_ep_len_mean");
timesteps = data(:, "time_total_timesteps");
episodes = data(:, "time_episodes");

%% Reshape data
rewards = table2array(rewards);
episodes = table2array(episodes);
length = table2array(length);
timesteps = table2array(timesteps);

%% Reward vs Episodes
plot(episodes, rewards)
hold on
plot(episodes, movmean(rewards, 2000), "Color", "Red")
ylabel('Mean Episode Reward')
xlabel('Number of Episodes')

%% Episode Length vs Episodes
plot(episodes, length, "Color", "#D95319")
hold on
plot(episodes, movmean(length, 2000), "Color", "Red")
ylabel('Episode Length')
xlabel('Number of Episodes')

%% Reward vs Timesteps
plot(timesteps, rewards)
ylabel('Mean Episode Reward')
xlabel('Number of Steps')
