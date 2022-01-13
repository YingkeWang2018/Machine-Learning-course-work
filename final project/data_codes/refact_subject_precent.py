import pandas as pd

def get_percentile_csv():
    '''
    get the csv file that has the precentage of each subject for each student
    '''
    train_df = pd.read_csv('../data/train_data.csv')
    subject_df = pd.read_csv('../data/subject_meta.csv')
    question_df = pd.read_csv('../data/question_meta.csv')
    student_df = pd.read_csv('../data/student_meta.csv')
    # get the empty result df
    subject_col = subject_df['subject_id']
    user_ids = student_df['user_id']
    result_cols = ['user_id'] + list(subject_col)
    data = []
    subject_none = [0.5] * len(subject_col) # fill each subject for each student with 0.5 accuary
    for user_id in user_ids:
        data.append([user_id] + subject_none)
    result_df = pd.DataFrame(data=data, columns=result_cols)
    user_groups = train_df.groupby(by=['user_id'], as_index=False)
    subject_col = list(subject_col)
    result_df[subject_col] = result_df.apply(lambda row: get_result_row(row, user_groups, question_df), axis=1, result_type='expand')
    # result_df.to_csv('subject_acc.csv', index=False)
    # get the train data user group
    return result_df


def get_result_row(row, user_groups, question_df):
    user_id = row['user_id']

    user_question_df = user_groups.get_group(user_id)
    accuracy_dict = {}
    user_question_df.apply(lambda row: fill_accuracy_dict(row, accuracy_dict, question_df), axis=1)
    for subject_id in accuracy_dict:
        subject_id_acc = accuracy_dict[subject_id]['correct']/accuracy_dict[subject_id]['total']
        row[subject_id] = subject_id_acc
    row = row[1: ]
    return row

def get_subject_list(subjects_str):
    """convert the '[1,2,3]' into a list of int"""
    subjects_str = subjects_str[1:-1]
    subject_lst = subjects_str.split(',')
    subject_lst = [int(subject) for subject in subject_lst]
    return subject_lst

def fill_accuracy_dict(row, accuracy_dict, question_df):
    question_id = row['question_id']
    is_correct = row['is_correct']
    question_sub = question_df.loc[question_df['question_id'] == question_id]
    subjects = question_sub['subject_id'].iloc[0]
    subjects = get_subject_list(subjects)

    for subject_id in subjects:
        if subject_id not in accuracy_dict:
            accuracy_dict[subject_id] = {'correct': 5, 'total': 10}
        if is_correct == 1:
            accuracy_dict[subject_id]['correct'] += 1
        accuracy_dict[subject_id]['total'] += 1

def get_question_sub_csv():
    question_df = pd.read_csv('../data/question_meta.csv')

    subject_df = pd.read_csv('../data/subject_meta.csv')
    # get the empty result df
    subject_col = subject_df['subject_id']
    result_cols = ['question_id'] + list(subject_col)
    subject_map = {}
    subject_none = [0] * len(subject_col)  # fill each subject for each student with 0.5 accuary
    question_ids = question_df['question_id']
    for question_id in question_ids:
        subject_map[question_id] = subject_none[:]
    question_df.apply(lambda row: get_question_subject(row, subject_map), axis=1)
    data = []
    for question_id in subject_map:
        data.append([question_id] + subject_map[question_id])
    result_df = pd.DataFrame(data=data, columns=result_cols)
    return result_df

def get_question_subject(row, subject_map):
    subject_lst = get_subject_list(row['subject_id'])
    question_id = row['question_id']
    for subject_id in subject_lst:
        subject_map[question_id][subject_id] = 1




if __name__ == '__main__':
    subject_acc_df = get_percentile_csv()