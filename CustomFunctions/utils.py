import pandas as pd





def get_consecutive_timepoints(
        df, #dataframe
        column: str, #string column to get consecutive timepoints from
        interval: int, #expected interval of "column"
        ):
    #sort the dataframe based on the column
    df_sorted = df.sort_values(column).reset_index(drop = True)
    #get differences over the column
    diff = df_sorted[column].diff()
    #create a list of all the places with time jumps starting with 0
    difflist = [0]
    difflist.extend(diff[diff>interval].index.to_list())
    if difflist[-1] < len(df_sorted):
        difflist.append(len(df_sorted))
    #make a list of lists with the indices of consecutive time points
    runs = [list(range(difflist[x], difflist[x+1])) for x in range(len(difflist)-1)]
    
    return df_sorted, runs
    