
## importation
import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

import matplotlib.pyplot as plt



def load_tuto_dataset():
    dataset = datasets.PROTEINS()
    graphs, graph_labels = dataset.load()
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )

    return(graphs, graph_labels)


def load_real_dataset():
    """
    """

    ## importation
    import pandas as pd
    import glob
    import craft_graph

    ## parameters
    roi_to_label = {}
    graph_list = []
    label_list = []

    ## load manifest & craft roi to label
    manifest = pd.read_csv("raw_data/OMIQ_metadata.csv")
    for index, row in manifest.iterrows():
        if(row['Lymphoma'] == "Lymphoma"):
            roi_to_label[row['Unnamed: 2']] = 2
        else:
            roi_to_label[row['Unnamed: 2']] = 1

    ## loop over roi data file
    cmpt_progress = 0
    total = len(glob.glob("discretized_data/*.csv"))
    for fcs_file in glob.glob("discretized_data/*.csv"):

        #-> extract roi name
        roi_name = fcs_file.split("/")
        roi_name = roi_name[-1]
        roi_name = roi_name.replace("_mean_normalized_discretized.csv", "")
        while(roi_name[-1] == "_"):
            roi_name = roi_name[:-1]

        #-> load roi data
        if(roi_name in roi_to_label.keys()):

            #--> create graph & update graph list
            graph = craft_graph.craft_graph(fcs_file)
            graph_list.append(graph)

            #--> add label to label list
            label = roi_to_label[roi_name]
            label_list.append(label)

        #-> display progress
        cmpt_progress +=1
        progress = (float(cmpt_progress) / float(total))*100.0
        print("[+][GRAPH-GENARATION] => "+str(progress)+"% {"+str(cmpt_progress)+"/"+str(total)+"}")


    ## return graph list and labels
    labels = pd.DataFrame(label_list)
    return(graph_list, labels)



def load_stupid_dataset():
    """
    """

    ## importation
    import pandas as pd
    import glob
    import craft_graph

    ## parameters
    roi_to_label = {}
    graph_list = []
    label_list = []

    ## load manifest & craft roi to label
    manifest = pd.read_csv("raw_data/OMIQ_metadata.csv")
    for index, row in manifest.iterrows():
        if(row['Lymphoma'] == "Lymphoma"):
            roi_to_label[row['Unnamed: 2']] = 2
        else:
            roi_to_label[row['Unnamed: 2']] = 1

    ## loop over roi data file
    cmpt_progress = 0
    total = len(glob.glob("stupid_data/*_discretized.csv"))
    for fcs_file in glob.glob("stupid_data/*_discretized.csv"):

        #-> extract label
        label = fcs_file.split("/")
        label = label[-1]
        label = label.split("_")
        label = label[1]
        if(label == "cat1"):
            label = 1
        elif(label == "cat2"):
            label = 2

        #--> create graph & update graph list
        graph = craft_graph.craft_graph(fcs_file)
        graph_list.append(graph)

        #--> add label to label list
        label_list.append(label)

        #-> display progress
        cmpt_progress +=1
        progress = (float(cmpt_progress) / float(total))*100.0
        print("[+][GRAPH-GENARATION] => "+str(progress)+"% {"+str(cmpt_progress)+"/"+str(total)+"}")


    ## return graph list and labels
    labels = pd.DataFrame(label_list)
    return(graph_list, labels)



## load dataset
dataset = load_real_dataset()
#dataset = load_tuto_dataset()
#dataset = load_stupid_dataset()
graphs = dataset[0]
graph_labels = dataset[1]
print("[+] Data extracted")

## format labels
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
print(graph_labels)

## Prepare graph generator
generator = PaddedGraphGenerator(graphs=graphs)


## craft model
k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 1]
dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)

x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
)

## Train the model
train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
)

gen = PaddedGraphGenerator(graphs=graphs)

train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 100

history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)

sg.utils.plot_history(history)
plt.savefig("images/history.png")

## Test model
output_file = open("model_results.txt", "w")
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
output_file.write("est Set Metrics:\n")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
    output_file.write("\t{}: {:0.4f}\n".format(name, val))
output_file.close()
