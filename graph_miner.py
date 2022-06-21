


def extract_connected_nodes(edges_file):
    """
    """

    ## importation
    import pandas as pd

    ## parameters
    subgraph_to_node = {}
    subgraph_cmpt = 0

    ## load data
    df = pd.read_csv(edges_file)

    ## loop over edges
    for index, row in df.iterrows():

        #-> extract information
        source = row["source"]
        target = row["target"]

        #-> check the other subgraph
        extend_existing_subgraph = False
        for subgraph in subgraph_to_node.keys():
            node_list = subgraph_to_node[subgraph]
            if(source in node_list and target not in node_list):
                subgraph_to_node[subgraph].append(target)
                extend_existing_subgraph = True
            if(target in node_list and source not in node_list):
                subgraph_to_node[subgraph].append(source)
                extend_existing_subgraph = True

        #-> create new subgraph
        if(not extend_existing_subgraph):
            subgraph_cmpt+=1
            subgraph = "subgraph_"+str(subgraph_cmpt)
            subgraph_to_node[subgraph] = [source,target]

    ## return subgraph_to_node
    return subgraph_to_node


def convert_subgraph_to_pattern_list(subgraph_to_node, nodes_file, variables_to_keep):
    """
    """

    ## importation
    import pandas as pd

    ## parameters
    node_to_pattern = {}
    pattern_list = []

    ## load nodes file
    df = pd.read_csv(nodes_file)
    df = df[variables_to_keep]

    ## loop over node
    node_cmpt = 0
    for index, row in df.iterrows():

        #-> identify node
        node_cmpt +=1
        node_name = "cell_"+str(node_cmpt)

        #-> craft pattern
        pattern = ""
        for k in list(row.keys()):
            scalar = row[k]
            pattern+=str(int(scalar))+"-"
        pattern = pattern[:-1]

        #-> update node to features
        node_to_pattern[node_name] = pattern

    ## loop over subgraph
    for subgraph in subgraph_to_node:
        node_list = subgraph_to_node[subgraph]
        pattern = []
        for node in node_list:
            pattern.append(node_to_pattern[node])
        pattern_list.append(pattern)

    #-> replace node with pattern
    return pattern_list




def craft_subgraph_to_pattern_list(subgraph_to_node, nodes_file, variables_to_keep):
    """
    """

    ## importation
    import pandas as pd

    ## parameters
    node_to_pattern = {}
    subgraph_to_pattern_list = {}

    ## load nodes file
    df = pd.read_csv(nodes_file)
    df = df[variables_to_keep]

    ## loop over node
    node_cmpt = 0
    for index, row in df.iterrows():

        #-> identify node
        node_cmpt +=1
        node_name = "cell_"+str(node_cmpt)

        #-> craft pattern
        pattern = ""
        for k in list(row.keys()):
            scalar = row[k]
            pattern+=str(int(scalar))+"-"
        pattern = pattern[:-1]

        #-> update node to features
        node_to_pattern[node_name] = pattern

    ## loop over subgraph
    for subgraph in subgraph_to_node:
        node_list = subgraph_to_node[subgraph]
        pattern = []
        for node in node_list:
            pattern.append(node_to_pattern[node])
        subgraph_to_pattern_list[subgraph] = pattern

    #-> replace node with pattern
    return subgraph_to_pattern_list








def fptree_mining(pattern_list, min_support):
    """
    """

    ## importation
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth

    ## init transaction database
    te = TransactionEncoder()
    te_ary = te.fit(pattern_list).transform(pattern_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    ## mine
    frequent_pattern = fpgrowth(df, min_support=min_support, use_colnames=True)

    ## return frequent_pattern
    return frequent_pattern



def extract_frequent_pattern(edges_file, nodes_file, variables_to_keep, min_support):
    """
    """

    ## extract subgraph
    graph_to_node = extract_connected_nodes(edges_file)

    ## extract pattern
    pattern_list = convert_subgraph_to_pattern_list(graph_to_node, nodes_file, variables_to_keep)

    ## mine frequent pattern
    frequent_pattern = fptree_mining(pattern_list, min_support)

    ## return frequent pattern
    return frequent_pattern



def spot_discriminating_pattern(pattern_1, pattern_2):
    """
    """

    ## parameters
    pattern_spec_1 = []
    pattern_spec_2 = []

    ## find frequent pattern specific to set 1
    for p1 in list(pattern_1["itemsets"]):
        p1 = list(p1)
        specific = True
        for p2 in list(pattern_2["itemsets"]):
            p2 = list(p2)
            if(p1 == p2):
                specific = False
        if(specific):
            pattern_spec_1.append(p1)

    ## find frequent pattern specific to set 2
    for p2 in list(pattern_2["itemsets"]):
        p2 = list(p2)
        specific = True
        for p1 in list(pattern_1["itemsets"]):
            p1 = list(p1)
            if(p2 == p1):
                specific = False
        if(specific):
            pattern_spec_2.append(p2)

    ## return results
    return {"set1":pattern_spec_1,"set2":pattern_spec_2}



def spot_discriminating_pattern_between_two_fcs_list(list1, list2, variables_to_keep, min_support):
    """
    very heavy computational cost
    """

    ## parameters
    pattern_spec_1 = []
    pattern_spec_2 = []

    ## extract all pattern for p1
    pattern_list_1 = []
    for fcs_name in list1:
        edge_file = "graph/edges/"+str(fcs_name)
        node_file = "graph/nodes/"+str(fcs_name)

        ## extact pattern
        pattern = extract_frequent_pattern(
                edge_file,
                node_file,
                variables_to_keep,
                min_support
            )
        ## update list of pattern for set 1
        for p1 in list(pattern["itemsets"]):
            p1 = list(p1)
            if(p1 not in list(pattern_list_1)):
                pattern_list_1.append(p1)

    ## extract all patterns for p2
    pattern_list_2 = []
    for fcs_name in list2:
        edge_file = "graph/edges/"+str(fcs_name)
        node_file = "graph/nodes/"+str(fcs_name)

        ## extact pattern
        pattern = extract_frequent_pattern(
                edge_file,
                node_file,
                variables_to_keep,
                min_support
            )
        ## update list of pattern for set 1
        for p2 in list(pattern["itemsets"]):
            p2 = list(p2)
            if(p2 not in list(pattern_list_2)):
                pattern_list_1.append(p2)


    ## find frequent pattern specific to set 1
    for p1 in pattern_list_1:
        specific = True
        for p2 in pattern_list_2:
            if(p1 == p2):
                specific = False
        if(specific):
            pattern_spec_1.append(p1)

    ## find frequent pattern specific to set 2
    for p2 in pattern_list_2:
        specific = True
        for p1 in pattern_list_1:
            if(p2 == p1):
                specific = False
        if(specific):
            pattern_spec_2.append(p2)

    print(pattern_spec_1)
    print(pattern_spec_2)

    ## return results
    return {"set1":pattern_spec_1,"set2":pattern_spec_2}


def extract_frequent_pattern_from_list(edges_file_list, nodes_file_list, variables_to_keep, min_support):
    """
    """

    ## parameters
    all_patterns = []

    cmpt = 0
    for edge_file in edges_file_list:
        node_file = nodes_file_list[cmpt]

        ## extract subgraph
        graph_to_node = extract_connected_nodes(edge_file)

        ## extract pattern
        pattern_list = convert_subgraph_to_pattern_list(graph_to_node, node_file, variables_to_keep)

        ## update
        all_patterns += pattern_list
        cmpt+=1

    ## mine frequent pattern
    frequent_pattern = fptree_mining(all_patterns, min_support)

    ## return frequent pattern
    return frequent_pattern


def reduce_fcs_file(fcs_file, edges_file, nodes_file, pattern_list, variables_to_keep):
    """
    """

    ## importation
    import pandas as pd

    ## parameters
    subgraph_selected = []
    nodes_selected = []

    ## extract subgraph
    graph_to_node = extract_connected_nodes(edges_file)

    ## craft graph_to_pattern
    graph_to_pattern_list = craft_subgraph_to_pattern_list(graph_to_node, nodes_file, variables_to_keep)

    ## select subgraph that match allowed patterns
    for subgraph in graph_to_pattern_list.keys():
        candidate_pattern = graph_to_pattern_list[subgraph]
        for allowed_pattern in pattern_list:
            pattern_present = True
            for elt in candidate_pattern:
                if(elt not in allowed_pattern):
                    pattern_present = False
            if(pattern_present and subgraph not in subgraph_selected):
                subgraph_selected.append(subgraph)

    ## select node
    for subgraph in subgraph_selected:
        nodes = graph_to_node[subgraph]
        for n in nodes:
            if(n not in nodes_selected):
                nodes_selected.append(n)


    ## proceed to file reduction
    #-> load original fcs file
    df = pd.read_csv(fcs_file)
    #-> craft cell variables
    cmpt = 0
    node_var = []
    for index, row in df.iterrows():
        cmpt+=1
        cell_id = "cell_"+str(cmpt)
        node_var.append(cell_id)

    #-> add variable to dataframe
    df["ID"] = node_var

    #-> perform selection
    df = df[df["ID"].isin(nodes_selected)]

    #-> save file
    output_name = fcs_file.split("/")
    output_name = output_name[-1]
    output_name = "reduced_data/"+str(output_name)
    output_name = output_name.replace(".csv", "_reduced.csv")
    df.to_csv(output_name, index=False)





def run_dimension_reduction(manifest_file):
    """
    The main function
    """

    ## importation
    import pandas as pd

    ## parameters
    set_1_file_list = []
    set_2_file_list = []
    variable_to_keep = ["CD38","Vimentin","FLT3L","CD56"]
    min_support = 0.6

    ## load manifest
    df_manifest = pd.read_csv(manifest_file)

    ## extract the two target list
    for index, row in df_manifest.iterrows():
        fname = row["FILENAME"]
        label = row["LABEL"]
        if(str(label) == "1"):
            set_1_file_list.append(fname)
        elif(str(label) == "2"):
            set_2_file_list.append(fname)

    ## display someting
    print("[+]Target files loaded")

    ## deal with set 1
    edge_file_list = []
    node_file_list = []
    for target_file in set_1_file_list:

        #-> craft edge file list
        edge_file = target_file.replace("discretized_data/", "graph/edges/")
        edge_file_list.append(edge_file)

        #-> craft node file list
        node_file = target_file.replace("discretized_data/", "graph/nodes/")
        node_file_list.append(node_file)

    #-> extract frequent pattern
    set1 = extract_frequent_pattern_from_list(
            edge_file_list,
            node_file_list,
            variable_to_keep,
            min_support
        )

    ## display something
    print("[+] Frequent pattern extracted from set1")


    ## deal with set 2
    edge_file_list = []
    node_file_list = []
    for target_file in set_2_file_list:

        #-> craft edge file list
        edge_file = target_file.replace("discretized_data/", "graph/edges/")
        edge_file_list.append(edge_file)

        #-> craft node file list
        node_file = target_file.replace("discretized_data/", "graph/nodes/")
        node_file_list.append(node_file)

    #-> extract frequent pattern
    set2 = extract_frequent_pattern_from_list(
            edge_file_list,
            node_file_list,
            variable_to_keep,
            min_support
        )

    ## display something
    print("[+] Frequent pattern extracted from set2")

    ## select discriminating patterns
    set_to_spec_patterns = spot_discriminating_pattern(set1, set2)

    ## display something
    print("[+] Discriminating patterns identified")

    ## reduce fcs files - set1
    for target_file in set_1_file_list:
        edge_file = target_file.replace("discretized_data/", "graph/edges/")
        node_file = target_file.replace("discretized_data/", "graph/nodes/")
        pattern_list = set_to_spec_patterns["set1"]
        reduce_fcs_file(
            target_file,
            edges_file,
            nodes_file,
            pattern_list,
            variables_to_keep
        )

    ## reduce fcs files - set2
    for target_file in set_2_file_list:
        edge_file = target_file.replace("discretized_data/", "graph/edges/")
        node_file = target_file.replace("discretized_data/", "graph/nodes/")
        pattern_list = set_to_spec_patterns["set2"]
        reduce_fcs_file(
            target_file,
            edges_file,
            nodes_file,
            pattern_list,
            variables_to_keep
        )

    ## display something
    print("[+] Reduction performed")




## TEST SPACE
run_dimension_reduction("manifest_lymphoma_discretized.csv")
#spot_discriminating_pattern(pattern_1, pattern_2)
"""
spot_discriminating_pattern_between_two_fcs_list(
    ["slide11roi10_1_mean_normalized_discretized.csv","slide11roi10_1_mean_normalized_discretized.csv","slide11roi9_1_mean_normalized_discretized.csv"],
    ["Slide1ROI4_1_mean_normalized_discretized.csv","Slide1ROI4_2_mean_normalized_discretized.csv","Slide1ROI3_1_mean_normalized_discretized.csv"],
    ["CD38","Vimentin","FLT3L","CD56"],
    0.8
)

set1 = extract_frequent_pattern_from_list(
    ["graph/edges/Slide1ROI4_1_mean_normalized_discretized.csv","graph/edges/Slide1ROI4_2_mean_normalized_discretized.csv","graph/edges/Slide1ROI3_1_mean_normalized_discretized.csv"],
    ["graph/nodes/Slide1ROI4_1_mean_normalized_discretized.csv","graph/nodes/Slide1ROI4_2_mean_normalized_discretized.csv","graph/nodes/Slide1ROI3_1_mean_normalized_discretized.csv"],
    ["CD38","Vimentin","FLT3L","CD56"],
    0.6
)

set2 = extract_frequent_pattern_from_list(
    ["graph/edges/slide11roi10_1_mean_normalized_discretized.csv","graph/edges/slide11roi10_1_mean_normalized_discretized.csv","graph/edges/slide11roi9_1_mean_normalized_discretized.csv"],
    ["graph/nodes/slide11roi10_1_mean_normalized_discretized.csv","graph/nodes/slide11roi10_1_mean_normalized_discretized.csv","graph/nodes/slide11roi9_1_mean_normalized_discretized.csv"],
    ["CD38","Vimentin","FLT3L","CD56"],
    0.6
)

stuff =  spot_discriminating_pattern(set1, set2)

fcs_file = "discretized_data/slide11roi10_1_mean_normalized_discretized.csv"
edges_file = "graph/edges/slide11roi10_1_mean_normalized_discretized.csv"
nodes_file = "graph/nodes/slide11roi10_1_mean_normalized_discretized.csv"
pattern_list = [['0-0-0-0', '0-0-0-0']]
variables_to_keep = ["CD38","Vimentin","FLT3L","CD56"]

reduce_fcs_file(
    fcs_file,
    edges_file,
    nodes_file,
    pattern_list,
    variables_to_keep
)
"""
