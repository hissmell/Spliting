import json
import matplotlib.pyplot as plt

with open("C:/Users/82102/VScodeProjects/Spliting/Spliting/Spliting/outputs_K10M4N70/feature_weights.json","r") as f:
    weights = json.load(f)

x = list(weights.keys())
y = list(weights.values())

sorted_data = sorted(zip(x, y), key=lambda item: item[1], reverse=False)
sorted_x, sorted_y = zip(*sorted_data)

plt.barh(sorted_x, sorted_y)
plt.title('iGAM')

plt.savefig("./feature_weights")