import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=3, suppress=True)

dataset_colu = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
dataset = pd.read_csv("magic04.data",names=dataset_colu).drop(["class"],axis=1)

# dataset['class'] = (dataset["class"] == "g").astype(int)
print(dataset.head())

for label in dataset.columns[1:]:
    plt.scatter(dataset[label],dataset["fLength"])
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel("fLength")
    plt.show()