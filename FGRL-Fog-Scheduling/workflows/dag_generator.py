
import pandas as pd
import networkx as nx

def load_workflow_dataset(task_csv, dep_csv):

    tasks=pd.read_csv(task_csv)
    deps=pd.read_csv(dep_csv)

    G=nx.DiGraph()

    for _,row in tasks.iterrows():
        G.add_node(int(row['task_id']))

    for _,row in deps.iterrows():
        G.add_edge(int(row['parent_task']), int(row['child_task']))

    return G
