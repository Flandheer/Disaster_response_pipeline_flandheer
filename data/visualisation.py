import pandas as pd

def visualiseCategories(df):

    df = df.drop(columns = ["id","message",'original','genre'])
    df = df.astype(int)

    viz = df.sum().sort_values( ascending= False)

    return viz
