import matplotlib.pyplot as plt

with open('out.txt', 'r') as f:
    lines = f.readlines()
    x = []
    y = []
    for i in range(len(lines)):
        l = int(lines[i])
        x.append(i)
        y.append(l)

    plt.plot(x, y)
    plt.show()
