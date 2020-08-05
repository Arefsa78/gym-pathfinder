import matplotlib.pyplot as plt

dirs = ['big_interval', 'human_control_level', 'her_episode_4', 'human_spars', 'her_episode_4_sparse', 'her_future_4_sparse', 'human_maps_verti']
for d in dirs:
    f = open(d + '/' + 'test_successes').readlines()
    acc = []
    for a in f:
        acc.append(float(a))
    print(d, max(acc))
    plt.plot(acc)
plt.show()