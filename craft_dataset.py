


def craft_mean_dataset():
    """
    Exploit the fist layer of information
        Lazy & stupid but possibly enough

    transaction are roi
    variables are mean of of marquers (for each roi)
    """

    ## importation
    import pandas as pd
    import glob
    import numpy as np

    ## parameters
    roi_to_label = {}
    roi_to_marker_to_mean = {}
    marker_list = []
    output_file_name = "simple_mean_marker_dataset.csv"

    ## load manifest & craft roi to label
    manifest = pd.read_csv("raw_data/OMIQ_metadata.csv")
    for index, row in manifest.iterrows():
        roi_to_label[row['Unnamed: 2']] = row['Lymphoma']

    ## loop over roi data file
    for fcs_file in glob.glob("raw_data/*.csv"):

        #-> extract roi name
        roi_name = fcs_file.split("/")
        roi_name = roi_name[-1]
        roi_name = roi_name.replace("_mean.csv", "")
        while(roi_name[-1] == "_"):
            roi_name = roi_name[:-1]

        #-> load roi data
        if(roi_name in roi_to_label.keys()):

            #-> init structure
            roi_to_marker_to_mean[roi_name] = {}

            #-> compute mean for each marker
            df = pd.read_csv(fcs_file)
            for k in list(df.keys()):
                if(k not in ["centroid_X", "centroid_Y"]):
                    marker_mean = np.mean(df[k])

                    #-> update data structure
                    roi_to_marker_to_mean[roi_name][k] = marker_mean

                    #-> update marker list
                    if(k not in marker_list):
                        marker_list.append(k)

    ## save data
    output_data = open(output_file_name, "w")

    ## craft header
    header = "ID,"
    for marker in marker_list:
        header+=str(marker)+","
    header +="LABEL"
    output_data.write(header+"\n")
    for roi in roi_to_marker_to_mean.keys():

        #-> extract info
        label = roi_to_label[roi]
        markers_to_mean = roi_to_marker_to_mean[roi]

        #-> craft line
        line = roi+","
        for marker in marker_list:
            line+=str(markers_to_mean[marker])+","
        line+=label

        #-> write line
        output_data.write(line+"\n")

    ## close file
    output_data.close()



def stupid_data_generator():
    """
    """

    ## importation
    import pandas as pd
    import random

    ## parameters
    nb_file_to_generate = 10
    nb_cell_in_file = 10
    header = ["centroid_X","centroid_Y","Vimentin","Machin","Truc","Cheesecake"]

    ## CLASS 1
    for f in range(0,nb_file_to_generate):
        #-> add specific graph for this class
        vec1 = [4,4,random.randint(0,10),random.randint(0,5),random.randint(0,5), random.randint(40,60)]
        vec2 = [4,5,random.randint(0,10),random.randint(0,5),random.randint(0,5), random.randint(40,60)]
        vec3 = [5,4,random.randint(0,10),random.randint(0,5),random.randint(0,5), random.randint(40,60)]
        matrix = [vec1,vec2,vec3]
        coordinates_list = ["4_4","4_5","5_4"]

        #-> fill the file
        for cell in range(0,nb_cell_in_file):

            ## generate coordinates
            coordinates_checked = False
            while(not coordinates_checked):
                x = random.randint(0,50)
                y = random.randint(0,50)
                coordinates = str(x)+"_"+str(y)
                if(coordinates not in coordinates_list):
                    coordinates_checked = True
                    coordinates_list.append(coordinates)

            ## craft vector
            vector = [x,y,random.randint(0,10),random.randint(0,5),random.randint(0,5), random.randint(0,10)]
            matrix.append(vector)

        ## save data
        df = pd.DataFrame(matrix, columns=header)
        df.to_csv("stupid_data/stupid_cat1_"+str(f)+".csv", index=False)


    ## CLASS 2
    for f in range(0,nb_file_to_generate):
        #-> add specific graph for this class
        vec1 = [1,1,random.randint(60,90),random.randint(0,5),random.randint(0,5), random.randint(0,10)]
        vec2 = [1,2,random.randint(60,90),random.randint(0,5),random.randint(0,5), random.randint(0,10)]
        vec3 = [2,1,random.randint(60,90),random.randint(0,5),random.randint(0,5), random.randint(0,10)]
        matrix = [vec1,vec2,vec3]
        coordinates_list = ["1_1","1_2","2_1"]

        #-> fill the file
        for cell in range(0,nb_cell_in_file):

            ## generate coordinates
            coordinates_checked = False
            while(not coordinates_checked):
                x = random.randint(0,50)
                y = random.randint(0,50)
                coordinates = str(x)+"_"+str(y)
                if(coordinates not in coordinates_list):
                    coordinates_checked = True
                    coordinates_list.append(coordinates)

            ## craft vector
            vector = [x,y,random.randint(0,10),random.randint(0,5),random.randint(0,5), random.randint(0,10)]
            matrix.append(vector)

        ## save data
        df = pd.DataFrame(matrix, columns=header)
        df.to_csv("stupid_data/stupid_cat2_"+str(f)+".csv", index=False)



def normalize_dataset():
    """
    IN PROGRESS
    """

    ## importation
    import pandas as pd
    import numpy as np
    import glob

    ## parameters
    marker_to_scalar = {}
    marker_to_mean = {}
    marker_to_std = {}
    marker_list = []

    ## identify target files & markers
    target_files = glob.glob("stupid_data/*.csv")
    df = pd.read_csv(target_files[0])
    for k in list(df.keys()):
        if(k not in ["centroid_X", "centroid_Y"]):
            marker_list.append(k)
            marker_to_scalar[k] = []

    ## Part 1 -  get the mean and std for all markers
    for tf in target_files:

        #-> load data
        df = pd.read_csv(tf)

        #-> loop over data & get scalars for each markers
        for index, row in df.iterrows():
            for k in list(row.keys()):
                if(k in marker_list):
                    marker_to_scalar[k].append(row[k])

    ## compute mean and std
    for k in marker_to_scalar.keys():
        vector = marker_to_scalar[k]
        marker_to_mean[k] = np.mean(vector)
        marker_to_std[k] = np.std(vector)

    ## Part 2 - Apply standardization
    ## loop over fcs file
    for tf in target_files:

        #-> load dataframe
        df = pd.read_csv(tf)

        #-> apply standardization
        for marker in marker_list:
            df[marker] = ((df[marker] - marker_to_mean[marker]) / marker_to_std[marker])

        #-> save dataframe to normalize file
        output_name = tf.replace(".csv", "_normalized.csv")
        df.to_csv(output_name, index=False)





def simple_discretization():
    """
    """

    ## importation
    import pandas as pd
    import glob

    ## parameters

    ## loop over target files
    for tf in glob.glob("stupid_data/*_normalized.csv"):

        #-> init new matrix
        matrix = []

        #-> load target files
        df = pd.read_csv(tf)

        #-> get header
        header = list(df.keys())

        #-> discretize
        for index, row in df.iterrows():
            vector = []
            for k in list(row.keys()):
                if(k in ["centroid_X", "centroid_Y"]):
                    vector.append(row[k])
                else:
                    scalar = row[k]
                    new_scalar = "NA"

                    #--> discretize
                    if(scalar < 0.2):
                        new_scalar = 0
                    elif(scalar < 0.4):
                        new_scalar = 1
                    elif(scalar < 0.6):
                        new_scalar = 2
                    elif(scalar < 0.8):
                        new_scalar = 3
                    elif(scalar <= 1):
                        new_scalar = 4
                    else:
                        new_scalar = 5

                    # update vector
                    vector.append(new_scalar)

            # update matrix
            matrix.append(vector)

        ## craft and save csv
        df = pd.DataFrame(matrix, columns=header)
        output_name = tf.replace(".csv", "_discretized.csv")
        df.to_csv(output_name, index=False)









#stupid_data_generator()
#craft_mean_dataset()
#normalize_dataset()
simple_discretization()
