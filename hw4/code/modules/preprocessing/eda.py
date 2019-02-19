import pandas as pd
import numpy as np
from mxnet import nd
from preprocessing import *

train = pd.read_csv("./data/raw/kaggle_house_pred_train.csv")
test_features = pd.read_csv("./data/raw/kaggle_house_pred_test.csv")

# Remove response variable from train
train_features = train[list(train)[:-1]]
# Concatenate the feature matrices
all_features = pd.concat([train_features, test_features], axis=0)
all_features = all_features.reset_index(drop=True)
# To help with easier do, replace all NULL value by np.nan
all_features = all_features.fillna(np.nan)

# List of colnames with null values
nullc = [c_name for c_name in list(train) if train[c_name].isnull().values.any()]

# Grouping features!

# EXTERIOR
# MSSubClass:
#   needs to convert to categorical!
update_dtypes = {"MSSubClass": str}
all_features = all_features.astype(update_dtypes)
# MSZoning
# LotFrontage:
#   Contains null values; most reasonably filled by 0
all_features.LotFrontage = all_features.LotFrontage.fillna(0)
# LotArea
# Street
# Alley:
#   Contains null values. Best left alone and let get_dummies()
#   list null values by itself

# LotShape
# LandContour
# Utilities
# LotConfig
# LandSlope
# Neighborhood
# ...
# OverallQual, OverallCond
# MasVnrType:
#   Contains null values indicating if there is a Masonry Veneer
#   MasVnrType is null iff MasVnrArea is null
#   This will be resolved by:
#       NaN in MasVnrTYpe will be converted into "None"
#       NaN in MasVnrType will be converted into 0
all_features.MasVnrType = all_features.MasVnrType.fillna("None")
all_features.MasVnrArea = all_features.MasVnrArea.fillna(0)

# ExterQual, ExterCond:
#   Ex, Gd, TA, Fa, Po espectively
#   9, 7, 5, 3, 2
verbal_rating_dict = {"Ex": np.float64(9),
                    "Gd": np.float64(7),
                    "TA": np.float64(5),
                    "Fa": np.float64(3),
                    "Po": np.float64(2),
                    "NA": np.nan}
ext_qual_num = [verbal_rating_dict[token] for
                token in all_features.ExterQual.values]
ext_cond_num = [verbal_rating_dict[token] for
                token in all_features.ExterCond.values]
all_features.ExterQual = ext_qual_num
all_features.ExterCond = ext_cond_num

# BASEMENT
# BsmtQual
#   Contains null values
#   Is null iff BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2,
#   BsmtFinSF1, BsmtFinSF2, BsmtUnSF, and TotalBsmtSF are null
#
# PROBLEM!
# There are rows in which BsmtQual is not NULL, but BsmtCond is NULL
# such NULL values will be replaced by the mode of BsmtCond of buildings
# from the same year built
row_indices = all_features.query("not BsmtQual.isnull() and BsmtCond.isnull()").index
for row_index in row_indices:
    # Get the year built
    YB = all_features.iloc[row_index].YearBuilt
    # Find the mode of BsmtCond
    BC_mode = all_features.query("YearBuilt == @YB").BsmtCond.mode()[0]
    # replace!
    all_features.iloc[row_index, list(all_features).index("BsmtCond")] = BC_mode

#   There are tows in which BsmtQual is not NULL but BsmtCond is NULL
#   such NULL values will be replaced by the mode of BsmtQual of buildings
#   from the same year built
row_indices = all_features.query("BsmtQual.isnull() and not BsmtCond.isnull()").index
for row_index in row_indices:
    # Get the year built
    YB = all_features.iloc[row_index].YearBuilt
    # Find the mode of BsmtCond
    BQ_mode = all_features.query("YearBuilt == @YB").BsmtQual.mode()[0]
    # replace!
    all_features.iloc[row_index, list(all_features).index("BsmtQual")] = BQ_mode

# BsmtCond being NULL implies:
#       BsmtExposure is NULL
#       BsmtFinType1 is NULL
#       BsmtFinSF1 is 0.0
# Basically BsmtFinSF1 should not have NaN values. Replace NaN with 0
all_features.BsmtFinSF1 = all_features.BsmtFinSF1.fillna(0)
#       BsmtFinType2 is NULL
#       BsmtFinSF2 is 0.0
# Basically BsmtFinSF2 should not have NaN values. Replace NaN with 0
all_features.BsmtFinSF2 = all_features.BsmtFinSF2.fillna(0)
#       BsmtUnfSF is 0.0 AND TotalBsmtSF is 0.0
all_features.BsmtUnfSF = all_features.BsmtUnfSF.fillna(0)
all_features.TotalBsmtSF = all_features.TotalBsmtSF.fillna(0)

# Now we convert BsmtQual and BsmtCond into numerical columns
# According to the dictionary above. NA will be converted to np.nan
# to be partitioned later
def vr_map(verbal_rating):
    # If verbal rating is string, return appropriate value
    if isinstance(verbal_rating, str):
        return verbal_rating_dict[verbal_rating]
    else:
        return np.nan
all_features.BsmtQual = all_features.BsmtQual.apply(vr_map)
all_features.BsmtCond = all_features.BsmtCond.apply(vr_map)


# INTERNAL
# Heating, HeatingQC, CentralAir are all just fine
# There is 1 row that has Electrical being NULL. Discard this row
# I can discard this row
drop_rows = list(all_features.query("Electrical.isnull()").index)
E_mode = all_features.Electrical.mode()[0]
all_features.iloc[drop_rows, list(all_features).index("Electrical")] = E_mode

# KitchenAbvGr: there is always a kitchen
# KitchenQual: same as ExterQual
#   There is 1 row with a NULL value for Kitchen Qual :(
#   but I cannot remove the row since it's in the test cases
#   Will replace it with lowest score possible
all_features.KitchenQual = all_features.KitchenQual.fillna("Po")
all_features.KitchenQual = all_features.KitchenQual.apply(vr_map)

# FIREPLACE
# Fireplaces
#   Indicates the number of fireplaces
#   Does not have NULL values
#   Is 0 iff FireplaceQu is NULL
#   Should be a partition condition
all_features.FireplaceQu = all_features.FireplaceQu.apply(vr_map)

# GarageType
#   contains NULL
#   is NULL iff
#           GarageYrBlt is NULL
#               There exists case in which GarageType is not NULL
#               but GarageYrBlt is not NULL
#               In this case, replace by YearBuilt
indices = list(all_features.query("not GarageType.isnull() and GarageYrBlt.isnull()").index)
GYB_index = list(all_features).index('GarageYrBlt')
YB_index = list(all_features).index('YearBuilt')
for row_index in indices:
    all_features.iloc[row_index, GYB_index] = all_features.iloc[row_index, YB_index]
#           GarageFinish is NULL
#               There exists case in which GarageType is not NULL
#               but GarageFinish is null.
#               In this case, replace by the mode of GarageFinish
#               from all garages from the same year
indices = list(all_features.query("not GarageType.isnull() and GarageFinish.isnull()").index)
target_col = list(all_features).index('GarageFinish')
for row_index in indices:
    #   Get GarageYrBlt
    GYB = all_features.iloc[row_index, GYB_index]
    GF_mode = all_features.query("GarageYrBlt == @GYB").iloc[:, target_col].mode()[0]
    all_features.iloc[row_index, target_col] = GF_mode

#           GarageCars, GarageArea is 0
#               There exists rows in which GarageType is not NULL
#               but GarageCars is NULL
#               Replace by the mode from the same year
indices = list(all_features.query("not GarageType.isnull() and GarageCars.isnull()").index)
target_col = list(all_features).index('GarageCars')
for row_index in indices:
    #   Get GarageYrBlt
    GYB = all_features.iloc[row_index, GYB_index]
    GC_mode = all_features.query("GarageYrBlt == @GYB").iloc[:, target_col].mode()[0]
    all_features.iloc[row_index, target_col] = GC_mode
#               There exists rows in which GarageType is not NULL
#               but GarageArea is NULL
#               Replace by the mean from the same year
indices = list(all_features.query("not GarageType.isnull() and GarageArea.isnull()").index)
target_col = list(all_features).index('GarageArea')
for row_index in indices:
    #   Get GarageYrBlt
    GYB = all_features.iloc[row_index, GYB_index]
    GA_mean = all_features.query("GarageYrBlt == @GYB").iloc[:, target_col].mean()
    all_features.iloc[row_index, target_col] = GA_mean
#           GarageCond, GarageQual are NULL
#               There exists case in which GarageType is not NULL
#               but these two are NULL
#               But they are not a concern since I can replace by 0
all_features.GarageCond = all_features.GarageCond.apply(vr_map)
all_features.GarageQual = all_features.GarageQual.apply(vr_map)
#   Should be a partition condition

# PoolArea
#   PoolArea == 0 implies PoolQC being NULL
#   But there are rows in which PoolArea > 0 but PoolQC is NULL
#   This will be handled by:
#   convert verbal rating into numerical rating, leaving NULL being NULL
#   Compute the mean of such numerical ratings as subsitute
all_features.PoolQC = all_features.PoolQC.apply(vr_map)
sub_val = all_features.PoolQC.mean()
row_indices = list(all_features.query("PoolArea > 0 and PoolQC.isnull()").index)
all_features.iloc[row_indices, list(all_features).index("PoolQC")] = sub_val
#   Should be a partition condition


# Others
#   Fence:
#       Contains null, but SalePrice mean is not differed by tooooooo much
#       Should be find doing a one-hot with dummyna
#   MiscFeature:
#       MiscFeature contains NULL values
#       almost all NULL values imply MiscVal == 0
#           There exists 1 row in which MiscFeature is NULL but MiscVal > 0
#           replace by NULL by "Othr"
row_indices = all_features.query("MiscFeature.isnull() and MiscVal > 0").index
all_features.iloc[row_indices,list(all_features).index("MiscFeature")] = "Othr"
#       But there exists 2 rows that have non-NULL value
#       for MiscFeature but a MiscVal of 0
#       I don't think it matters

# Year and Month sold
# The earlier of the years is 2006, so I will integrate the two columsn into
# a single time stamp by
all_features["SaleTimestamp"] = (all_features.YrSold - 2006)*12 + all_features.MoSold
# then YrSold and MoSold can be dropped
all_features = all_features.drop(["YrSold", "MoSold"], axis=1)

# Convert to one hot!
cat_colnames = list(all_features.dtypes[all_features.dtypes == 'object'].index)
num_colnames = list(all_features.dtypes[all_features.dtypes != 'object'].index)
for cat_colname in cat_colnames:
    all_features = make_one_hot(all_features,
        list(all_features).index(cat_colname))

# Normalize!
# There a just a few more columns still having NULL values
nullc = [c_name for c_name in list(all_features) if all_features[c_name].isnull().values.any()]
for colname in nullc:
    # If you are one of the columns below, replace NaN by 0
    if colname in ["BsmtQual", "BsmtCond",
                    "FireplaceQu",
                    "GarageQual", "GarageCond",
                    "PoolQC"]:
        all_features[colname] = all_features[colname].fillna(0)
    # Otherwise, replace by mean of the column
    else:
        col_mean = all_features[colname].mean()
        all_features[colname] = all_features[colname].fillna(col_mean)

print([c_name for c_name in list(all_features) if all_features[c_name].isnull().values.any()])

for num_colname in num_colnames:
    col_mean = all_features[num_colname].mean()
    col_se = all_features[num_colname].std()
    all_features[num_colname] = (all_features[num_colname] - col_mean) / col_se

# Convert DataFrame (with all numerical columns!) to numpy array
# but drop the Id!
train_X = all_features.iloc[:train.shape[0], 1:].values
test_X = all_features.iloc[train.shape[0]:, 1:].values
train_Y = train.SalePrice.values.reshape((-1, 1))

np.save("./data/clean/train_X.npy", train_X)
np.save("./data/clean/test_X.npy", test_X)
np.save("./data/clean/train_Y.npy", train_Y)
