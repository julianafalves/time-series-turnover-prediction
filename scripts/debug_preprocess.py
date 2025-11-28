from src.preprocessing import preprocess_data
import pandas as pd

p='data/Time_Series___Turnover (1).csv'
df=pd.read_csv(p)
X_train, X_test, y_train, y_test, pipeline, col_names = preprocess_data(df, mode='ts', n_lags=6)
print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print('y_train shape', y_train.shape)
print('col_names', col_names[:20])
print('Sample y_train', y_train[:5])
print('Pipeline numeric feature count:', len(pipeline.named_steps['preprocessor'].transformers_[0][2]))
