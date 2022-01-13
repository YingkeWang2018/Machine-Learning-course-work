import pandas as pd
import numpy as np
import math
from datetime import datetime

def prepro_main():
    student_df = pd.read_csv('../data/student_meta.csv')

    # student_df.info()

    def cal_age(x):
        if type(x) == str:
            return int(2021 - datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f').year)

    def aju_age(x):
        if math.isnan(x) or x < 10:
            return ((student_df['age'].mean()) + 1) / (student_df['age'].max())
        else:
            return x / (student_df['age'].max())

    def adj_pp(x):
        if math.isnan(x):
            return student_df['premium_pupil'].mean()
        return x

    student_df["age"] = student_df["data_of_birth"].map(cal_age)

    student_df["age_adj"]=student_df["age"].map(aju_age)
    student_df["premium_pupil_adj"]=student_df['premium_pupil'].map(adj_pp)
    #
    student_results = student_df[["user_id", "gender","age", "age_adj", "premium_pupil_adj"]]
    median_age = student_results['age'].median()
    print(f'the median age of data set is {median_age}')
    student_results = student_results.sort_values(by=['user_id'])
    return student_results
    # student_results.to_csv("student_features.csv",index=False)
if __name__ == '__main__':
    pass