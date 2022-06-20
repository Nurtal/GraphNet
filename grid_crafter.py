

def craft_grid(fcs_file, target_marker):
    """
    """

    ## importation
    import pandas as pd
    import numpy as np

    ## parameters
    max_x = -1
    max_y = -1
    coordinates_to_value = {}

    ## load fcs file
    df = pd.read_csv(fcs_file)

    ## extract coordinates
    for index, row in df.iterrows():

        #-> extract information
        x = row["centroid_X"]
        y = row["centroid_Y"]
        value = row[target_marker]

        #-> update grid information
        key = str(int(x))+"_"+str(int(y))
        coordinates_to_value[key] = value

        #-> update max
        if(x > max_x):
            max_x = x
        if(y > max_y):
            max_y = y

    ## craft matrix
    matrix = []
    for y in range(0,int(max_y)+1):
        vector = []
        for x in range(0,int(max_x)+1):
            coordinates = str(x)+"_"+str(y)
            if(coordinates in coordinates_to_value):
                scalar = coordinates_to_value[coordinates]
            else:
                scalar = 0
            vector.append(scalar)
        vector = np.array(vector)
        matrix.append(vector)
    matrix = np.array(matrix)

    ## return matrix
    return matrix



#craft_grid("raw_data/Slide1Roi1_2_mean.csv", "Vimentin")
craft_grid("test_dataset.csv", "Vimentin")
