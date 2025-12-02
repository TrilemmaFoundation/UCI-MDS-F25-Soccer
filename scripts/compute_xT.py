'''
Utils for computing xThreat.
'''
import sqlite3
import ast
import pandas as pd
import numpy as np

def get_grid(loc, n_rows:int=8, n_cols:int=12):
    '''
    Return the row, column for the location based on a 120x80 soccer pitch (helper function for get_transition_probas)

    Parameters
    ----------
    loc : list of float
        The location to classify. Typically in the form (x, y).
    n_rows : int
        The number of rows the pitch is split into (splits the 80 dimension)
    n_cols : int
        The number of columns the pitch is split into (spltis the 120 dimension)
    '''
    # note: the -1 is so that we don't index outside
    row = int((loc[1]  * (n_rows-1)) / 80)
    col = int((loc[0] * (n_cols-1)) / 120)

    return [row, col]

def get_transition_probas(db_path:str='../statsbomb/statsbomb_euro2020.db', n_rows:int=8, n_cols:int=12):
    '''
    Computes transition probabilities based on the database with an events table.

    Parameters
    ----------
    db_path : str
        The path to the database with the events table. Default `../statsbomb/statsbomb_euro2020.db`
    n_rows : int
        The number of rows to break the 80x120 pitch into (breaks the 80 part)
    n_cols : int
        The number of columns to break the 80x120 pitch into (breaks the 120 part)

    Returns
    -------
    transition_rel_freqs : numpy.Array
        A n_rows x n_cols x n_rows x n_cols numpy.Array object of transition probabilities. 
        Specifically, the probability of moving from `(x1, y1)` to `(x2, y2)` is located at 
        `transition_rel_freqs[y1, x1, y2, x2]` (the x and y are switched because we do rows then columns).
    shot_rel_freqs : numpy.Array
        A n_rows x n_cols numpy.Array object of shot probabilities. The probability of shooting from `(x, y)` is 
        located at `shot_rel_freqs[y, x]` (again, x and y are switched because we index row then column).
    '''

    conn = sqlite3.connect(db_path)

    # grab relevant data
    # passes
    passes = pd.read_sql_query('select location, pass_end_location from events where type = "Pass"', conn)
    # carries
    carries = pd.read_sql_query('select location, carry_end_location from events where type = "Carry"', conn)
    # shots
    shots = pd.read_sql_query('select location from events where type = "Shot"', conn)

    conn.close()

    # convert pass/carry/shot stuff into proper lists (originally strings)
    pass_starts = passes['location'].map(ast.literal_eval).tolist()
    pass_ends = passes['pass_end_location'].map(ast.literal_eval).tolist()

    carry_starts = carries['location'].map(ast.literal_eval).tolist()
    carry_ends = carries['carry_end_location'].map(ast.literal_eval).tolist()

    shot_locs = shots['location'].map(ast.literal_eval).tolist()

    # covert locations to grid locations
    pass_starts_grid = [get_grid(loc, n_rows, n_cols) for loc in pass_starts]
    pass_ends_grid = [get_grid(loc, n_rows, n_cols) for loc in pass_ends]
    carry_starts_grid = [get_grid(loc, n_rows, n_cols) for loc in carry_starts]
    carry_ends_grid = [get_grid(loc, n_rows, n_cols) for loc in carry_ends]
    shot_locs_grid = [get_grid(loc, n_rows, n_cols) for loc in shot_locs]

    # count transitions and shots
    transition_freqs = np.zeros((n_rows, n_cols, n_rows, n_cols))
    shot_freqs = np.zeros((n_rows, n_cols))

    for i, (start) in enumerate(pass_starts_grid):
        end = pass_ends_grid[i]
        transition_freqs[start[0], start[1], end[0], end[1]] += 1

    for i, start in enumerate(carry_starts_grid):
        end = carry_ends_grid[i]
        transition_freqs[start[0], start[1], end[0], end[1]] += 1

    for loc in shot_locs_grid:
        shot_freqs[loc[0], loc[1]] += 1

    # convert transitions/shots to relative frequencies
    transition_rel_freqs = np.zeros((n_rows, n_cols, n_rows, n_cols))
    shot_rel_freqs = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            total_n = np.sum(transition_freqs[i,j,:,:]) + shot_freqs[i,j]
            if total_n == 0:
                continue
            # get relative frequency
            transition_rel_freqs[i,j,:,:] = transition_freqs[i,j,:,:] / total_n
            shot_rel_freqs[i,j] = shot_freqs[i,j] / total_n

    return transition_rel_freqs, shot_rel_freqs



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from PIL import Image

    DB_PATH = "../statsbomb/statsbomb_euro2020.db"
    n_rows = 8*2
    n_cols = 12*2

    transition_rel_freqs, shot_rel_freqs = get_transition_probas(db_path=DB_PATH,
                                                                 n_rows=n_rows,
                                                                 n_cols=n_cols)
    # nice formatting
    img_path = "../imgs/soccer-field.jpg"
    img = Image.open(img_path)

    # make relative frequency NaN when it's zero
    transition_rel_freqs[transition_rel_freqs == 0] = np.nan

    plt.imshow(img, extent=[0, n_cols, 0, n_rows])
    plt.imshow(transition_rel_freqs[10, 16, :, :], cmap=mpl.colormaps["Greens"], alpha=0.8)

    plt.xlim(0, n_cols)
    plt.ylim(0, n_rows)
    plt.show()