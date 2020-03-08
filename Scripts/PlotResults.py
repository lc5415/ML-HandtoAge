import pandas as pd
import matplotlib.pyplot as plt

arch1 = pd.read_csv('Results/TrainTestLoss1.csv')
arch2 = pd.read_csv('Results/TrainTestLoss2.csv')
arch3 = pd.read_csv('Results/TrainTestLoss3.csv')

plt.plot(arch1["Test loss"], label = "Arch 1")
plt.plot(arch2["Test loss"], label = "Arch 2")
plt.plot(arch3["Test loss"], label = "Arch 3")
plt.legend()
plt.grid()
plt.show()
