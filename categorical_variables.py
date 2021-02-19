import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = pd.read_csv('iowa_train.csv', index_col='Id')
X_test = pd.read_csv('iowa_test.csv', index_col='Id')
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
test_cols_with_missing = [col for col in X_test.columns if X_test[col].isnull().any()]
object_cols = [col for col in X.columns if X[col].dtype == 'object']
object_cols_with_missing_nd = pd.merge(pd.DataFrame(object_cols), pd.DataFrame(cols_with_missing), how='inner').values.tolist()
object_cols_with_missing = []
for col in object_cols_with_missing_nd:
    object_cols_with_missing.append(''.join(col))
test_object_cols_with_missing_nd = pd.merge(pd.DataFrame(object_cols), pd.DataFrame(test_cols_with_missing), how='inner').values.tolist()
test_object_cols_with_missing = []
for col in test_object_cols_with_missing_nd:
    test_object_cols_with_missing.append(''.join(col))
low_cardinality_cols = [col for col in object_cols if X[col].nunique()<10]
high_cardinality_cols = [col for col in object_cols if X[col].nunique()>=10]
missing_count_X = X[object_cols_with_missing].isnull().sum()
cols_alot_missing = missing_count_X[missing_count_X>700].index
cols_some_missing = missing_count_X[missing_count_X<700]
cols_some_missing = cols_some_missing[cols_some_missing>0].index

X.drop(cols_alot_missing, axis=1, inplace=True)
X_test.drop(cols_alot_missing, axis=1, inplace=True)
for col in cols_alot_missing:
    test_object_cols_with_missing.remove(col)
    low_cardinality_cols.remove(col)

mfreq_imputer = SimpleImputer(strategy = 'most_frequent')
X_object_imputed = pd.DataFrame(mfreq_imputer.fit_transform(X[cols_some_missing]))
X_object_imputed.columns = cols_some_missing
X_object_imputed.index = X.index
X_without_imputed_cols = X.drop(cols_some_missing, axis=1)
X_all =  pd.concat([X_object_imputed, X_without_imputed_cols], axis=1)

X_test_object_imputed = pd.DataFrame(mfreq_imputer.transform(X_test[cols_some_missing]))
X_test_object_imputed.columns = cols_some_missing
X_test_object_imputed.index = X_test.index
X_test_without_imputed_cols = X_test.drop(cols_some_missing, axis=1)
#X_test_without_imputed_cols = X_test_without_imputed_cols.drop(cols_alot_missing)
X_test_all =  pd.concat([X_test_object_imputed, X_test_without_imputed_cols], axis=1)

for col in cols_some_missing:
    if col in test_object_cols_with_missing:
        test_object_cols_with_missing.remove(col)
X_test_object_imputed = pd.DataFrame(mfreq_imputer.fit_transform(X_test_all[test_object_cols_with_missing]))
X_test_object_imputed.columns = test_object_cols_with_missing
X_test_object_imputed.index = X_test.index
X_test_without_imputed_cols = X_test_all.drop(test_object_cols_with_missing, axis=1)
X_test_all =  pd.concat([X_test_object_imputed, X_test_without_imputed_cols], axis=1)

aligned_X_all, aligned_X_test_all = X_all.align(X_test_all, join='left', axis=1)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_X_all = pd.DataFrame(OH_encoder.fit_transform(aligned_X_all[low_cardinality_cols]))
OH_X_test_all = pd.DataFrame(OH_encoder.transform(aligned_X_test_all[low_cardinality_cols]))
OH_X_all.index = X.index
OH_X_test_all.index = X_test.index
num_X_all = X_all.drop(low_cardinality_cols, axis=1)
num_X_test_all = X_test_all.drop(low_cardinality_cols, axis=1)
OH_X = pd.concat([OH_X_all, num_X_all], axis=1)
OH_X_test = pd.concat([OH_X_test_all, num_X_test_all], axis=1)

aligned_OH_X, aligned_OH_X_test = OH_X.align(OH_X_test, join='left', axis=1)

label_X = aligned_OH_X.copy()
label_X_test = aligned_OH_X_test.copy()
label_encoder = LabelEncoder()
for col in high_cardinality_cols:
    label_X[col] = label_encoder.fit_transform(label_X[col])
    label_X_test[col] = label_encoder.transform(label_X_test[col])

mean_imputer = SimpleImputer(strategy = 'mean')
X_processed = pd.DataFrame(mean_imputer.fit_transform(label_X))
X_processed.columns = label_X.columns
X_test_processed = pd.DataFrame(mean_imputer.transform(label_X_test))
X_test_processed.columns = label_X_test.columns

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, train_size=0.8,
                                                  test_size=0.2)
RF_model = RandomForestRegressor(n_estimators=100)
RF_model.fit(X_train, y_train)
preds = RF_model.predict(X_val)
mae = mean_absolute_error(preds, y_val)
print('mae = ', mae)

preds_test = RF_model.predict(X_test_processed)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
