# %%
import pandas as pd

# %%
df_app = pd.read_csv('applicant_data.csv', index_col=0)
df_emp = pd.read_csv('employer_data.csv', index_col=0)

# %%
# drop invalid data

# drop data with gender == Other
df_app = df_app[df_app['gender'] != 'Other']
df_emp = df_emp[df_emp['gender'] != 'Other']

# drop data with too many understanding attempts
def valid(row):
    return all(row[x] < 3 for x in row.index if 'attempts' in x)

df_app = df_app[[valid(row) for _, row in df_app.iterrows()]]
df_emp = df_emp[[valid(row) for _, row in df_emp.iterrows()]]

# %%
# convert applicant data to form where it can be analyzed easily

self_eval_ratings = {
    'terrible': 1,
    'not good': 2,
    'neutral': 3,
    'good': 4,
    'very good': 5,
    'exceptional': 6
}
self_eval_key = {v: k for k, v in self_eval_ratings.items()}

credibility_ratings = {
    'not credible': 1,
    'somewhat not credible': 2,
    'somewhat credible': 3,
    'credible': 4,
}
credibility_key = {v: k for k, v in credibility_ratings.items()}

self_eval_statement = {
    '“I conduct all tasks assigned to me with the needed attention, and therefore I would work hard in a job that required me to perform well in tasks similar to the application questions.”': 1,
    '“Usually I am the best at what I do, and therefore I would succeed in a job that required me to perform well in tasks similar to the application questions.”': 2,
    'I prefer not to include either of these statements in my application.': 3,
}

df_app['treatment'] = df_app['treatment'].map(lambda x: x + 1).astype(int)

df_app['self_eval'] = df_app['self_eval'].apply(
    lambda x: self_eval_ratings[x]
)
df_app['self_eval_statement'] = df_app['self_eval_statement'].apply(
    lambda x: self_eval_statement[x]
)

df_app['credibility_of_100'] = df_app['credibility_of_100'].map(
    lambda x: credibility_ratings[x]
)

df_app['counterfactual_promote'] = df_app['counterfactual_promote'].map(
    lambda x: self_eval_ratings[x]
)

df_app['female'] = (df_app['gender'] == 'Female').astype(int)

df_app['avatar'] = df_app['avatar'].map(lambda x: x.rstrip('.jpg'))

df_app['bachelors_or_higher'] = df_app['education'].map(
    lambda x: x in ["Bachelor's Degree", "Master's Degree", "Ph.D. or higher"]
).astype(int)

df_app['grad_degree'] = df_app['education'].map(
    lambda x: x in ["Master's Degree", "Ph.D. or higher"]
).astype(int)

df_app['employed_fulltime'] = df_app['employed'].map(
    lambda x: x == 'Employed full-time'
)

def set_as_int(field):
    df_app[field] = df_app[field].astype(int)

for field in [
    'self_eval_agree',
    'age',
    'eval_correct',
    'noneval_correct',
]:
    set_as_int(field)

df_app.rename(
    columns = {
        'self_eval': 'promote1',
        'self_eval_agree': 'promote2',
        'self_eval_statement': 'promote3',
    },
    inplace = True
)

df_app['promote3_attentive'] = (df_app['promote3'] == 1).astype(int)
df_app['promote3_boastful'] = (df_app['promote3'] == 2).astype(int)

# %%
# convert employer data to form where it can be analyzed easily

agree_ratings = {
    'na': 0,
    'disagree strongly': 1,
    'disagree': 2,
    'agree': 3,
    'strongly agree': 4,
}
agree_ratings_key = {v: k for k, v in agree_ratings.items()}

confident_ratings = {
    'na': 0,
    'not confident': 1,
    'somewhat confident': 2,
    'appropriately confident': 3,
    'overly confident': 4,
}
confident_ratings_key = {v: k for k, v in confident_ratings.items()}

df_emp['female'] = (df_emp['gender'] == 'Female').astype(int)

df_emp['age'] = df_emp['age'].astype(int)

df_emp['bachelors_or_higher'] = df_emp['education'].map(
    lambda x: x in ["Bachelor's Degree", "Master's Degree", "Ph.D. or higher"]
).astype(int)

df_emp['grad_degree'] = df_emp['education'].map(
    lambda x: x in ["Master's Degree", "Ph.D. or higher"]
).astype(int)

df_emp['employed_fulltime'] = df_emp['employed'].map(
    lambda x: x == 'Employed full-time'
).astype(int)

for field in df_emp.columns:
    if '_agree' in field:
        df_emp[field] = df_emp[field].map(lambda x: 0 if pd.isna(x) else agree_ratings[x])
    elif '_confident' in field:
        df_emp[field] = df_emp[field].map(lambda x: 0 if pd.isna(x) else confident_ratings[x])


# %%
# helper functions
def split_to_int(string):
    return [int(x) for x in string.split('-')]

def split_to_float(string):
    return [float(x) for x in string.split('-')]

# %%
# create df of employer wage bids
bids_list = []


for i, row in df_emp.iterrows():
    emp_is_female = int(row['gender'] == 'Female')
    applicants = row['applicants'].split('-')
    bids = split_to_float(row['bids'])
    for j, (applicant, bid) in enumerate(zip(applicants, bids)):
        try:
            app_row = df_app.loc[applicant]
        except KeyError:
            # applicant was discarded
            continue
        promote_type_seen = 1 if j < 10 else 2 if j < 20 else 3
        bids_list.append({
            'employer': i,
            'applicant': applicant,
            'treatment': app_row['treatment'],
            'emp_is_female': emp_is_female,
            'app_is_female': app_row['female'],
            'promote_type_seen': promote_type_seen,
            'app_promote1': app_row['promote1'],
            'app_promote2': app_row['promote2'],
            'app_promote3_attentive': app_row['promote3_attentive'],
            'app_promote3_boastful': app_row['promote3_boastful'],
            'app_eval_correct': app_row['eval_correct'],
            'bid': bid
        })

df_bids = pd.DataFrame(bids_list).set_index('employer')

# %%
# add treatment field to df_emp
df_emp['treatment'] = df_emp.index.map(lambda x: df_bids.loc[x]['treatment'].iloc[0])

# %%
df_bids.to_csv('employer_wage_bids.csv')

variable_labels = {
    'applicant': 'id of applicant being bid on',
    'treatment': 'treatment group for employer and applicant',
    'emp_is_female': 'employer\'s gender, 1 iff employer is female',
    'app_is_female': 'applicant\'s gender, 1 iff applicant is female',
    'promote_type_seen': 'the self-promotion type seen by the employer in making this wage bid',
    'app_promote_1': 'applicant\'s choice for first self-promotion type',
    'app_promote_2': 'applicant\'s choice for second self-promotion type',
    'app_promote_3_attentive': 'whether applicant chose attentive self-description for third self-promotion type',
    'app_promote_3_boastful': 'whether applicant chose boastful self-description for third self-promotion type',
    'app_eval_correct': 'number of application questions answered correctly by applicant',
    'bid': 'employer\'s wage bid for this applicant',
}

value_labels = {
    'treatment': {
        1: 'only self-promotion revealed',
        2: 'self-promotion and gender revealed',
        3: 'self-promotion, gender, and performance revealed',
    },
    'promote_type_seen': {
        1: 'first self-promotion type (1-6 scale)',
        2: 'second self-promotion type (0-100 agreement indicator)',
        3: 'third self-promotion type (statement)',
    },
    'app_promote1': self_eval_key,
}

df_bids.to_stata('employer_wage_bids.dta', variable_labels = variable_labels, value_labels = value_labels)

# %%
# create df of applicant wage guesses
wage_guesses = []

for i, row in df_app.iterrows():
    treatment = row['treatment']
    guesser_is_female = int(row['gender'] == 'Female')
    other_performance = split_to_int(row['wage_guess_perform'])
    promote_type_seen = [x + 1 for x in split_to_int(row['wage_guess_promote_type'])]
    other_promote1 = split_to_int(row['wage_guess_promote1'])
    other_promote2 = split_to_int(row['wage_guess_promote2'])
    other_promote3 = [x + 1 for x in split_to_int(row['wage_guess_promote3'])]
    other_is_female = [
        int(x == 'Female') for x in row['wage_guess_gender'].split('-')
    ]
    wage_guess = split_to_float(row['wage_guess_other'])
    perform_guess = row['perform_guess_other'] if pd.isna(row['perform_guess_other']) else split_to_int(row['perform_guess_other'])
    approp_guess = row['approp_guess_other'] if pd.isna(row['approp_guess_other']) else split_to_int(row['approp_guess_other'])

    for i, (performance_, promote_type_, promote1_, promote2_, promote3_, other_is_female_, wage_guess_) in enumerate(zip(
        other_performance, promote_type_seen, other_promote1, other_promote2, other_promote3, other_is_female, wage_guess
    )):
        wage_guesses.append({
            'guesser': i,
            'treatment': treatment,
            'guesser_is_female': guesser_is_female,
            'other_is_female': other_is_female_,
            'promote_type_seen': promote_type_,
            'other_promote1': promote1_,
            'other_promote2': promote2_,
            'other_promote3_attentive': int(promote3_ == 1),
            'other_promote3_boastful': int(promote3_ == 2),
            'other_eval_correct': performance_,
            'wage_guess': wage_guess_,
            'perform_guess': perform_guess[i] if isinstance(perform_guess, list) else perform_guess,
            'approp_guess': approp_guess[i] + 1 if isinstance(approp_guess, list) else approp_guess,
        })

df_guesses = pd.DataFrame(wage_guesses).set_index('guesser')

# %%
df_guesses.to_csv('applicant_wage_guesses.csv')

variable_labels = {
    'treatment': 'treatment group for guesser',
    'guesser_is_female': 'guesser\'s gender, 1 iff guesser is female',
    'other_is_female': 'other\'s gender, 1 iff other is female',
    'promote_type_seen': 'the self-promotion type seen by the guesser in making this wage guess',
    'other_promote_1': 'other\'s choice for first self-promotion type',
    'other_promote_2': 'other\'s choice for second self-promotion type',
    'other_promote_3_attentive': 'whether other chose attentive self-description for third self-promotion type',
    'other_promote_3_boastful': 'whether other chose boastful self-description for third self-promotion type',
    'other_eval_correct': 'number of application questions answered correctly by other',
    'wage_guess': 'wage guess for the other',
    'perform_guess': "guess of other's performance on the job performance questions",
    "approp_guess": "guess of employers' evals of the social appropriateness of other's responses",
}

value_labels = {
    'treatment': {
        1: 'only self-promotion revealed',
        2: 'self-promotion and gender revealed',
        3: 'self-promotion, gender, and performance revealed',
    },
    'promote_type_seen': {
        1: 'first self-promotion type (1-6 scale)',
        2: 'second self-promotion type (0-100 agreement indicator)',
        3: 'third self-promotion type (statement)',
    },
    'other_promote1': self_eval_key,
    'approp_guess': {
        1: "Very socially inappropriate",
        2: "Socially inappropriate",
        3: "Somewhat socially inappropriate",
        4: "Somewhat socially appropriate",
        5: "Socially appropriate",
        6: "Very socially appropriate",
    }
}

df_guesses.to_stata('applicant_wage_guesses.dta', variable_labels = variable_labels, value_labels = value_labels)

# %%
# prune unnecessary columns from df_app
df_app = df_app[[
    'treatment',
    'age',
    'female',
    'bachelors_or_higher',
    'grad_degree',
    'employed_fulltime',
    'eval_correct',
    'noneval_correct',
    'avatar',
    'promote1',
    'promote2',
    'promote3_attentive',
    'promote3_boastful',
    'study_topic_guess',
    'male_avg_answers_guess',
    'female_avg_answers_guess',
    'credibility_of_100',
    'counterfactual_promote',
    'self_promote_reason',
]]

df_app.index.rename('applicant', inplace = True)

# %%
# export to csv

df_app.to_csv('applicant_data_clean.csv')


# add labels, then export to stata

value_labels = {
    'treatment': 'treatment group for applicant',
    'age': 'applicant\'s age in years',
    'female': '1 iff applicant is female',
    'bachelors_or_higher': '1 iff applicant has a bachelor\'s, masters, PhD or higher',
    'grad_degree': '1 iff applicant has a masters, PhD or higher',
    'employed_fulltime': '1 iff applicant is employed full-time',
    'eval_correct': 'number of application questions answered correctly by applicant (seen by employer)',
    'noneval_correct': 'number of job performance questions answered correctly by applicant (used for employer bonus)',
    'avatar': 'identifier for the gendered avatar assigned to the applicant',
    'promote_1': 'applicant\'s choice for first self-promotion type',
    'promote_2': 'applicant\'s choice for second self-promotion type',
    'promote_3_attentive': 'whether applicant chose attentive self-description for third self-promotion type',
    'promote_3_boastful': 'whether applicant chose boastful self-description for third self-promotion type',
    'study_topic_guess': 'applicant\'s guess for the topic of the study',
    'male_avg_answers_guess': 'applicant\'s guess for average number of correctly answered job performance questions among all male applicants',
    'female_avg_answers_guess': 'applicant\'s guess for average number of correctly answered job performance questions among all female applicants',
    'credibility_of_100': 'applicant\'s evaluation of credibility of another applicant choosing 100 (out of 100) for second self-promotion type',
    'counterfactual_promote': 'how applicant would have self-promoted if their gender had been revealed (for treatment 1) or not revealed (for treatments 2 and 3)',
    'self_promote_reason': 'applicant\'s reason for their answer for the first self-promotion type',
}

value_labels = {
    'treatment': {
        1: 'only self-promotion revealed',
        2: 'self-promotion and gender revealed',
        3: 'self-promotion, gender, and performance revealed',
    },
    'promote1': self_eval_key,
    'counterfactual_promote': self_eval_key,
    'credibility_of_100': credibility_key,
}

df_app.applymap(
    # remove non latin-1 characters
    lambda x: x if not isinstance(x, str) else x.encode('latin-1', 'namereplace').decode('latin-1')
).to_stata('applicant_data_clean.dta', variable_labels = variable_labels, value_labels = value_labels)

# %%
# prune columns from df_emp

df_emp = df_emp[[
    'treatment',
    'age',
    'female',
    'bachelors_or_higher',
    'grad_degree',
    'employed_fulltime',
    'study_topic_guess',
    'male_avg_answers_guess',
    'female_avg_answers_guess',
    'exit_survey_female_avatar',
    'exit_survey_male_avatar',
    'exit_survey_perform',
    'exit_survey_promote',
    'male_enjoy_agree',
    'male_respect_agree',
    'male_approachable_agree',
    'male_interpersonal_agree',
    'male_recommend_agree',
    'male_confident_describe',
    'female_enjoy_agree',
    'female_respect_agree',
    'female_approachable_agree',
    'female_interpersonal_agree',
    'female_recommend_agree',
    'female_confident_describe',
]]

df_emp.index.rename('employer', inplace = True)

# %%
# export to csv

df_emp.to_csv('employer_data_clean.csv')


# add labels, then export to stata

value_labels = {
    'treatment': 'treatment group for employer',
    'age': 'employer\'s age in years',
    'female': '1 iff employer is female',
    'bachelors_or_higher': '1 iff employer has a bachelor\'s, masters, PhD or higher',
    'grad_degree': '1 iff employer has a masters, PhD or higher',
    'employed_fulltime': '1 iff employer is employed full-time',
    'study_topic_guess': 'employer\'s guess for the topic of the study',
    'male_avg_answers_guess': 'employer\'s guess for average number of correctly answered job performance questions among all male applicants',
    'female_avg_answers_guess': 'employer\'s guess for average number of correctly answered job performance questions among all female applicants',
    'exit_survey_female_avatar': 'identifier for hypothetical female avatar used in exit survey',
    'exit_survey_male_avatar': 'identifier for hypothetical male avatar used in exit survey',
    'exit_survey_perform': '(randomly generated) performance score for hypothetical applicants in exit survey, out of 10, represents applicant\'s performance on application questions',
    'exit_survey_promote': '(randomly generated) self-promotion statement chosen by hypothetical applicants in exit survey, represents applicant\'s choice for first self-promotion type',
    'male_enjoy_agree': 'employer\'s agreement with statement "I would enjoy working with him" about the hypothetical male applicant',
    'male_respect_agree': 'employer\'s agreement with statement "He would treat me with respect" about the hypothetical male applicant',
    'male_approachable_agree': 'employer\'s agreement with statement "He would be approachable for an issue that bothered me" about the hypothetical male applicant',
    'male_interpersonal_agree': 'employer\'s agreement with the statement "He has strong interpersonal skills" about the hypothetical male applicant',
    'male_recommend_agree': 'employer\'s agreement with the statement "I would recommend him as a colleague to others" about the hypothetical male applicant',
    'male_confident_describe': 'employer\'s description of the confidence of the hypothetical male applicant',
    'female_enjoy_agree': 'employer\'s agreement with statement "I would enjoy working with her" about the hypothetical female applicant',
    'female_respect_agree': 'employer\'s agreement with statement "She would treat me with respect" about the hypothetical female applicant',
    'female_approachable_agree': 'employer\'s agreement with statement "She would be approachable for an issue that bothered me" about the hypothetical female applicant',
    'female_interpersonal_agree': 'employer\'s agreement with the statement "She has strong interpersonal skills" about the hypothetical female applicant',
    'female_recommend_agree': 'employer\'s agreement with the statement "I would recommend her as a colleague to others" about the hypothetical female applicant',
    'female_confident_describe': 'employer\'s description of the confidence of the hypothetical female applicant',
}

value_labels = {
    'treatment': {
        1: 'only self-promotion revealed',
        2: 'self-promotion and gender revealed',
        3: 'self-promotion, gender, and performance revealed',
    },
    'exit_survey_promote': self_eval_key, 
    'male_enjoy_agree': agree_ratings_key,
    'male_respect_agree': agree_ratings_key,
    'male_approachable_agree': agree_ratings_key,
    'male_interpersonal_agree': agree_ratings_key,
    'male_recommend_agree': agree_ratings_key,
    'male_confident_describe': confident_ratings_key,
    'female_enjoy_agree': agree_ratings_key,
    'female_respect_agree': agree_ratings_key,
    'female_approachable_agree': agree_ratings_key,
    'female_interpersonal_agree': agree_ratings_key,
    'female_recommend_agree': agree_ratings_key,
    'female_confident_describe': confident_ratings_key,
}

df_emp.to_stata('employer_data_clean.dta', variable_labels = variable_labels, value_labels = value_labels)

