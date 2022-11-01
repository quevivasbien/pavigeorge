# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

from warnings import filterwarnings
filterwarnings('ignore')

# %%
df_bids = pd.read_csv('employer_wage_bids.csv', index_col='employer')
# df_bids.head()

# %%
print('test hypothesis 1 with first self-promotion type')
print('expect to see app_promote1 > 0')

data = df_bids[(df_bids['treatment'] == 1) & (df_bids['promote_type_seen'] == 1)]
X = sm.add_constant(
    data['app_promote1']
)
fitted = sm.OLS(data['bid'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary())

# %%
print('\n\n test hypothesis 1 with second self-promotion type')
print('expect to see app_promote2 > 0')

data = df_bids[(df_bids['treatment'] == 1) & (df_bids['promote_type_seen'] == 2)]
X = sm.add_constant(
    data['app_promote2']
)
fitted = sm.OLS(data['bid'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary())

# %%
print('\n\ntest hypothesis 2 with first self-promotion type')
print('expect to see app_promote1 > 0 and app_is_female*app_promote1 < 0')

data = df_bids[(df_bids['treatment'] == 2) & (df_bids['promote_type_seen'] == 1)]
X = sm.add_constant(
    np.stack(
        (
            data['app_promote1'],
            data['app_is_female'],
            data['app_is_female'] * data['app_promote1']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['bid'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ['const', 'app_promote1', 'app_is_female', 'app_is_female*app_promote1']
))

# %%
print('\n\ntest hypothesis 2 with second self-promotion type')
print('expect to see app_promote2 > 0 and app_is_female*app_promote2 < 0')

data = df_bids[(df_bids['treatment'] == 2) & (df_bids['promote_type_seen'] == 2)]
X = sm.add_constant(
    np.stack(
        (
            data['app_promote2'],
            data['app_is_female'],
            data['app_is_female'] * data['app_promote2']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['bid'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ['const', 'app_promote2', 'app_is_female', 'app_is_female*app_promote2']
))

# %%
print('\n\ntest hypothesis 3 with first self-promotion type')
print('expect to see app_promote1 > 0 and app_is_female*app_promote1 < 0')

data = df_bids[(df_bids['treatment'] == 3) & (df_bids['promote_type_seen'] == 1)]
X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['app_promote1'],
                    data['app_is_female'],
                    data['app_is_female'] * data['app_promote1']
                ),
                axis=1
            ),
            pd.get_dummies(data['app_eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['bid'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = (
        ["const", "app_promote1", "app_is_female", "app_is_female*app_promote1"]
        + [f"fe{i}" for i in range(X.shape[1] - 4)]
    )
))

# %%
print('\n\ntest hypothesis 3 with second self-promotion type')
print('expect to see app_promote2 > 0 and app_is_female*app_promote2 < 0')

data = df_bids[(df_bids['treatment'] == 3) & (df_bids['promote_type_seen'] == 1)]
X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['app_promote2'],
                    data['app_is_female'],
                    data['app_is_female'] * data['app_promote2']
                ),
                axis=1
            ),
            pd.get_dummies(data['app_eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['bid'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = (
        ["const", "app_promote2", "app_is_female", "app_is_female*app_promote2"]
        + [f"fe{i}" for i in range(X.shape[1] - 4)]
    )
))

# %%
df_guesses = pd.read_csv('applicant_wage_guesses.csv', index_col='guesser')
# df_guesses.head()

# %%
print('\n\ntest hypothesis 4 with first self-promotion type')
print('expect to see other_promote1 > 0 and guesser_is_female*other_promote1 < 0')

data = df_guesses[(df_guesses['treatment'] == 1) & (df_guesses['promote_type_seen'] == 1)]
X = sm.add_constant(
    np.stack(
        (
            data['other_promote1'],
            data['guesser_is_female'],
            data['guesser_is_female'] * data['other_promote1']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ["const", "other_promote1", "guesser_is_female", "guesser_is_female*other_promote1"]
))

# %%
print('\n\ntest hypothesis 4 with second self-promotion type')
print('expect to see other_promote2 > 0 and guesser_is_female*other_promote2 < 0')

data = df_guesses[(df_guesses['treatment'] == 1) & (df_guesses['promote_type_seen'] == 2)]
X = sm.add_constant(
    np.stack(
        (
            data['other_promote2'],
            data['guesser_is_female'],
            data['guesser_is_female'] * data['other_promote2']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ["const", "other_promote2", "guesser_is_female", "guesser_is_female*other_promote2"]
))

# %%
print('\n\ntest hypothesis 5 with first self-promotion type and female guessers only')
print('expect to see other_promote1 > 0 and other_is_female*other_promote1 < 0')

data = df_guesses[(df_guesses['treatment'] == 2) & (df_guesses['promote_type_seen'] == 1) & df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.stack(
        (
            data['other_promote1'],
            data['other_is_female'],
            data['other_is_female'] * data['other_promote1']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ["const", "other_promote1", "other_is_female", "other_is_female*other_promote1"]
))

# %%
print('\n\ntest hypothesis 5 with first self-promotion type and male guessers only')
print('expect to see other_promote1 > 0 and other_is_female*other_promote1 < 0')

data = df_guesses[(df_guesses['treatment'] == 2) & (df_guesses['promote_type_seen'] == 1) & ~df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.stack(
        (
            data['other_promote1'],
            data['other_is_female'],
            data['other_is_female'] * data['other_promote1']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ["const", "other_promote1", "other_is_female", "other_is_female*other_promote1"]
))

# %%
print('\n\ntest hypothesis 5 with second self-promotion type and female guessers only')
print('expect to see other_promote2 > 0 and other_is_female*other_promote2 < 0')

data = df_guesses[(df_guesses['treatment'] == 2) & (df_guesses['promote_type_seen'] == 2) & df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.stack(
        (
            data['other_promote2'],
            data['other_is_female'],
            data['other_is_female'] * data['other_promote2']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ["const", "other_promote2", "other_is_female", "other_is_female*other_promote2"]
))

# %%
print('\n\ntest hypothesis 5 with second self-promotion type and male guessers only')
print('expect to see other_promote2 > 0 and other_is_female*other_promote2 < 0')

data = df_guesses[(df_guesses['treatment'] == 2) & (df_guesses['promote_type_seen'] == 2) & ~df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.stack(
        (
            data['other_promote2'],
            data['other_is_female'],
            data['other_is_female'] * data['other_promote2']
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = ["const", "other_promote2", "other_is_female", "other_is_female*other_promote2"]
))

# %%
print("\ntest hypothesis 6 with first self-promotion type and female guessers only")
print("expect to see other_promote1 > 0 and other_is_female*other_promote1 < 0")

data = df_guesses[(df_guesses['treatment'] == 3) & (df_guesses['promote_type_seen'] == 1) & df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['other_promote1'],
                    data['other_is_female'],
                    data['other_is_female'] * data['other_promote1']
                ),
                axis=1
            ),
            pd.get_dummies(data['other_eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = (
        ["const", "other_promote1", "other_is_female", "other_is_female*other_promote1"]
        + [f"fe{i}" for i in range(X.shape[1] - 4)]
    )
))

# %%
print("\n\ntest hypothesis 6 with first self-promotion type and male guessers only")
print("expect to see other_promote1 > 0 and other_is_female*other_promote1 < 0")

data = df_guesses[(df_guesses['treatment'] == 3) & (df_guesses['promote_type_seen'] == 1) & ~df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['other_promote1'],
                    data['other_is_female'],
                    data['other_is_female'] * data['other_promote1']
                ),
                axis=1
            ),
            pd.get_dummies(data['other_eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = (
        ["const", "other_promote1", "other_is_female", "other_is_female*other_promote1"]
        + [f"fe{i}" for i in range(X.shape[1] - 4)]
    )
))

# %%
print("\n\ntest hypothesis 6 with second self-promotion type and female guessers only")
print("expect to see other_promote2 > 0 and other_is_female*other_promote2 < 0")

data = df_guesses[(df_guesses['treatment'] == 3) & (df_guesses['promote_type_seen'] == 2) & df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['other_promote2'],
                    data['other_is_female'],
                    data['other_is_female'] * data['other_promote2']
                ),
                axis=1
            ),
            pd.get_dummies(data['other_eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = (
        ["const", "other_promote2", "other_is_female", "other_is_female*other_promote2"]
        + [f"fe{i}" for i in range(X.shape[1] - 4)]
    )
))

# %%
print("\n\ntest hypothesis 6 with second self-promotion type and male guessers only")
print("expect to see other_promote2 > 0 and other_is_female*other_promote2 < 0")

data = df_guesses[(df_guesses['treatment'] == 3) & (df_guesses['promote_type_seen'] == 2) & ~df_guesses['guesser_is_female']]
X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['other_promote2'],
                    data['other_is_female'],
                    data['other_is_female'] * data['other_promote2']
                ),
                axis=1
            ),
            pd.get_dummies(data['other_eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['wage_guess'], X).fit(
    cov_type='cluster', cov_kwds={'groups': data.index}
)
print(fitted.summary(
    xname = (
        ["const", "other_promote2", "other_is_female", "other_is_female*other_promote2"]
        + [f"fe{i}" for i in range(X.shape[1] - 4)]
    )
))

# %%
df_app = pd.read_csv("applicant_data_clean.csv", index_col="applicant")
# df_app.head()

# %%
print('\n\ntest hypothesis 7 with first self-promotion type')
print('expect to see female < 0')

X = sm.add_constant(
    np.concatenate(
        (
            df_app['female'].to_numpy().reshape(-1, 1),
            pd.get_dummies(df_app['eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(df_app['promote1'], X).fit(
    cov_type='HC1'
)
print(fitted.summary(
    xname = ["const", "female"] + [f'fe{i}' for i in range(X.shape[1] - 2)]
))

# %%
print('\n\ntest hypothesis 7 with second self-promotion type')
print('expect to see female < 0')

X = sm.add_constant(
    np.concatenate(
        (
            df_app['female'].to_numpy().reshape(-1, 1),
            pd.get_dummies(df_app['eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(df_app['promote2'], X).fit(
    cov_type='HC1'
)
print(fitted.summary(
    xname = ["const", "female"] + [f'fe{i}' for i in range(X.shape[1] - 2)]
))

# %%
print('\n\ntest hypothesis 8 with first self-promotion type')
print('expect to see treatment2 = 0 and treatment2*female < 0')

data = df_app[df_app['treatment'] != 3]

X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['female'],
                    data['treatment'] == 2,
                    (data['treatment'] == 2) & data['female']
                ),
                axis=1
            ),
            pd.get_dummies(data['eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['promote1'], X).fit(
    cov_type='HC1'
)
print(fitted.summary(
    xname = (
        ["const", "female", "treatment2", "treatment2*female"]
        + [f'fe{i}' for i in range(X.shape[1] - 4)]
    )
))

# %%
print('\n\ntest hypothesis 8 with second self-promotion type')
print('expect to see treatment2 = 0 and treatment2*female < 0')

data = df_app[df_app['treatment'] != 3]

X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['female'],
                    data['treatment'] == 2,
                    (data['treatment'] == 2) & data['female']
                ),
                axis=1
            ),
            pd.get_dummies(data['eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['promote2'], X).fit(
    cov_type='HC1'
)
print(fitted.summary(
    xname = (
        ["const", "female", "treatment2", "treatment2*female"]
        + [f'fe{i}' for i in range(X.shape[1] - 4)]
    )
))

# %%
print('\n\ntest hypothesis 9 with first self-promotion type')
print('expect to see treatment3 = 0 and treatment3*female < 0')

data = df_app[df_app['treatment'] != 1]

X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['female'],
                    data['treatment'] == 3,
                    (data['treatment'] == 3) & data['female']
                ),
                axis=1
            ),
            pd.get_dummies(data['eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['promote1'], X).fit(
    cov_type='HC1'
)
print(fitted.summary(
    xname = (
        ["const", "female", "treatment3", "treatment3*female"]
        + [f'fe{i}' for i in range(X.shape[1] - 4)]
    )
))

# %%
print('\n\ntest hypothesis 9 with second self-promotion type')
print('expect to see treatment3 = 0 and treatment3*female < 0')

data = df_app[df_app['treatment'] != 1]

X = sm.add_constant(
    np.concatenate(
        (
            np.stack(
                (
                    data['female'],
                    data['treatment'] == 3,
                    (data['treatment'] == 3) & data['female']
                ),
                axis=1
            ),
            pd.get_dummies(data['eval_correct']).to_numpy()[:, :-1]
        ),
        axis=1
    )
)
fitted = sm.OLS(data['promote2'], X).fit(
    cov_type='HC1'
)
print(fitted.summary(
    xname = (
        ["const", "female", "treatment3", "treatment3*female"]
        + [f'fe{i}' for i in range(X.shape[1] - 4)]
    )
))

# %%



