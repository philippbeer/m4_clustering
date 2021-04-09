"""
This contains needed utilities
"""


def get_top_n_indexes(val_list: list, n: int = 3) -> list:
    """
    returns the indexes of the highest values in the list
    Params:
    -------
    val_list : list of values from wich highest N are
                        to returned as index position
    Returns:
    -------
    top_indexes : list of indexes containing the highest N values
    """
    top_scores_sorted = sorted(val_list, reverse=True)
    top_indexes = [val_list.index(val) for val in top_scores_sorted[:n + 1]]
    return top_indexes
