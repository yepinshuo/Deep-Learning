import numpy as np
import pandas as pd

def normalize(df, col_idx, scale=True):
    # Given a DataFrame object and a column index that points to a
    # numerical column, return a NEW copy of the data frame
    # with that column normalized (if scale is True) or
    # mean centered (if scale is false)
    # If there are NA and NaNs, replace them with the mean; this means
    # computing the mean without taking into consideration these values
    # then assigning the appropriate rows with 0, since the output is
    # either mean centered or normalized

    # Get a copy of the data frame object and the column out of the
    # copies output DF. From now all operations will be done on the
    # copy
    output_df = df.copy()
    target_col = output_df.iloc[:, col_idx]

    # Compute the mean and the sample error (divided by n-1)
    col_mean = tagret_col.mean()
    col_se = target_col.std()
    # If scale is set to true, normalize the column to make
    # mean 0 and sample error 1. Otherwise, only do mean centering
    if scale:
        target_col = (target_col - col_mean) / col_se
    else:
        target_col = (target_col - col_mean)

    # Fill the NaNs and NAs by 0 because we have at least mean centered
    # the column
    target_col.fillna(0)
    # Fill the copy with the new column
    output_df.iloc[:, col_idx] = target_col
    return output_df

def make_one_hot(df, col_idx):
    # Given a DataFrame object and a column index that points to
    # a categorical variables. Return a NEW copy of the data frame
    # in which the categorical column is removed, and one-hot columns
    # are created and appended to the end

    # Get a copy of the data frame for output, get the column
    # then drop this column from the output DF
    output_df = df.copy()
    # When extracting the target column, enforce the dtype "str"
    target_col = output_df.iloc[:, col_idx].astype(str)
    # Extract the column name so that NaNs from different columns
    # will not get confused; this means dummy_na should be False
    target_colname = list(output_df)[col_idx]
    # Replace values within this column by colname_val
    target_col[:] = target_colname + "_" + target_col
    # Drop the categorical column
    output_df = output_df.drop(labels=list(output_df)[col_idx],
                            axis=1)

    # Use pd method to create one-hot, including using NAN as a distinct
    # label
    one_hot = pd.get_dummies(target_col, dummy_na = False)

    # Concatenate the two dfs and return it
    return pd.concat([output_df, one_hot], axis=1)

def equiv_null(df, col_idx_1, col_idx_2):
    # Given a DataFrame object, return the following three things:
    # Whether the rows on which both columns are NULL are exactly the same
    # the indices of rows on which col_1 is NULL
    # the indices of rows on which col_2 is NULL

    null_rows_1 = np.asarray(df.iloc[:, col_idx_1].isnull()).nonzero()[0]
    null_rows_2 = np.asarray(df.iloc[:, col_idx_2].isnull()).nonzero()[0]

    return set(null_rows_1) == set(null_rows_2), null_rows_1, null_rows_2
