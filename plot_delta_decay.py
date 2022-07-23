import seaborn as sns
import matplotlib.pyplot as plt

with open('jcr_app.log', 'r') as f:
    sweeps = []
    delta = []
    for line in f.readlines():
        elements = line.split(',')
        if len(elements) < 2:
            continue
        sweeps.append(float(elements[0].split(':')[1].strip()))
        delta.append(float(elements[1].split(':')[1].strip()))
g = sns.lineplot(x=sweeps, y=delta)
g.set(yscale='log')
plt.xlabel('sweep')
plt.ylabel('delta')
plt.hlines(0.01, 0, 30, colors=['r'])
plt.show()

