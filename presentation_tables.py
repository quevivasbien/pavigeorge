## Meant to be run with `python presentation_tables.py > tables/tables.tex`

# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

# %%
df_bids = pd.read_csv('employer_wage_bids.csv', index_col='employer')
df_guesses = pd.read_csv('applicant_wage_guesses.csv', index_col='guesser')
df_app = pd.read_csv('applicant_data_clean.csv', index_col='applicant')

# %%
def make_table(fitted_models, title=None):
    tables = []
    for name, fitted in fitted_models.items():
        tables.append(
            pd.DataFrame(
                np.stack((fitted.params, fitted.bse, fitted.pvalues), axis=1),
                index=fitted.params.index,
                columns = pd.MultiIndex.from_arrays([
                    [name, name, name],
                    ['coef.', '(s.e.)', 'p']
                ]),
            )
        )
    df = pd.concat(tables, axis=1)
    df = df.loc[[i for i in df.index if not i.startswith('fe')]]
    sty = df.style.format(
        subset=pd.IndexSlice[:, pd.IndexSlice[:, 'coef.']],
        formatter=lambda x: f'{x:.2g}',
        na_rep = ''
    ).format(
        subset=pd.IndexSlice[:, pd.IndexSlice[:, '(s.e.)']],
        formatter=lambda x: f'({x:.2g})',
        na_rep = ''
    ).format(
        subset=pd.IndexSlice[:, pd.IndexSlice[:, 'p']],
        formatter=lambda x: f'{x:.2f}',
        na_rep = ''
    ).format_index(
        escape="latex", axis=0
    )
    
    return sty.to_latex(
        column_format = 'l' + '|'.join(['rlc']*len(fitted_models)),
        multicol_align='p{12em}',
        hrules = True,
        caption = title,
        position_float='centering',
    )

# %%
def hyp1_3_table(promote_type = 1):
    data = df_bids[
        (df_bids['treatment'] == 1) & (df_bids['promote_type_seen'] == promote_type)
    ]
    X = sm.add_constant(
        data[f'app_promote{promote_type}']
    )
    X.columns=['const', 'Self-promotion']

    fitted1 = sm.OLS(data['bid'], X).fit(
        cov_type='cluster', cov_kwds={'groups': data.index}
    )

    data = df_bids[
        (df_bids['treatment'] == 2) & (df_bids['promote_type_seen'] == promote_type)
    ]
    X = pd.DataFrame(
        sm.add_constant(
            np.stack(
                (
                    data[f'app_promote{promote_type}'],
                    data['app_is_female'],
                    data['app_is_female'] * data[f'app_promote{promote_type}']
                ),
                axis=1
            )
        ),
        columns = ['const', 'Self-evaluation', 'Female', 'Self-evaluation x Female'],
        index = data.index
    )
    fitted2 = sm.OLS(data['bid'], X).fit(
        cov_type='cluster', cov_kwds={'groups': data.index}
    )

    data = df_bids[
        (df_bids['treatment'] == 3) & (df_bids['promote_type_seen'] == promote_type)
    ]
    X = pd.DataFrame(
        sm.add_constant(
            np.concatenate(
                (
                    np.stack(
                        (
                            data[f'app_promote{promote_type}'],
                            data['app_is_female'],
                            data['app_is_female'] * data[f'app_promote{promote_type}']
                        ),
                        axis=1
                    ),
                    pd.get_dummies(data['app_eval_correct']).to_numpy()[:, :-1]
                ),
                axis=1
            )
        ),
        columns = [
            'const', 'Self-evaluation', 'Female', 'Self-evaluation x Female',
        ] + [f"fe{i}" for i in range(9)],
        index = data.index
    )
    fitted3 = sm.OLS(data['bid'], X).fit(
        cov_type='cluster', cov_kwds={'groups': data.index}
    )

    return make_table(
        {
            'Self-evaluation': fitted1,
            'Self-evaluation + gender': fitted2,
            'Self-evaluation + gender + performance*': fitted3
        },
        title = f'Employer bids, with {"first" if promote_type == 1 else "second"} self-evaluation type',
    )
    

# %%
print(hyp1_3_table(1))

# %%
print(hyp1_3_table(2))

# %%
def get_hyp7_fit(treatment=None, promote_type=1):
    data = df_app if treatment is None else df_app[df_app['treatment'] == treatment]
    X = pd.DataFrame(
        sm.add_constant(
            np.concatenate(
                (
                    data['female'].to_numpy().reshape(-1, 1),
                    pd.get_dummies(data['eval_correct']).to_numpy()[:, :-1]
                ),
                axis=1
            )
        ),
        columns = ['const', 'Female'] + [f"fe{i}" for i in range(9)],
        index = data.index
    )
    return sm.OLS(data[f'promote{promote_type}'], X).fit(
        cov_type='HC1'
    )

def hyp7_table(promote_type=1):
    fits = [get_hyp7_fit(t, promote_type) for t in [None, 1, 2, 3]]
    return make_table(
        {
            'All treatments': fits[0],
            'Self-evaluation': fits[2],
            'Self-evaluation + gender': fits[2],
            'Self-evaluation + gender + performance*': fits[3]
        },
        title = f'Applicant self-evaluation, with {"first" if promote_type == 1 else "second"} self-evaluation type',
    )

# %%
print(hyp7_table(1))

# %%
print(hyp7_table(2))

# %%
def hyp4_fit(promote_type=1):
    data = df_guesses[(df_guesses['treatment'] == 1) & (df_guesses['promote_type_seen'] == promote_type)]
    X = pd.DataFrame(
            sm.add_constant(
            np.stack(
                (
                    data[f'other_promote{promote_type}'],
                    data['guesser_is_female'],
                    data['guesser_is_female'] * data[f'other_promote{promote_type}']
                ),
                axis=1
            )
        ),
        columns = ['const', 'Self-evaluation', 'Female guesser', 'Self-evaluation x Female guesser'],
        index = data.index
    )
    return sm.OLS(data['wage_guess'], X).fit(
        cov_type='cluster', cov_kwds={'groups': data.index}
    )

def hyp4_table():
    return make_table(
        {
            'First self-promotion type': hyp4_fit(1),
            'Second self-promotion type': hyp4_fit(2),
        },
        title = 'Wage guesses (self-evaluation-only treatment)'
    )

# %%
print(hyp4_table())

# %%
def hyp5_6_table(female = 1, promote_type = 1):
    data = df_guesses[
        (df_guesses['treatment'] == 2) & (df_guesses['promote_type_seen'] == promote_type) & (df_guesses['guesser_is_female'] == female)
    ]
    X = pd.DataFrame(
        sm.add_constant(
            np.stack(
                (
                    data[f'other_promote{promote_type}'],
                    data['other_is_female'],
                    data['other_is_female'] * data[f'other_promote{promote_type}']
                ),
                axis=1
            ),
        ),
        columns = ['const', 'Self-evaluation', 'Female', 'Self-evaluation x Female'],
        index = data.index
    )

    fitted1 = sm.OLS(data['wage_guess'], X).fit(
        cov_type='cluster', cov_kwds={'groups': data.index}
    )

    data = df_guesses[(df_guesses['treatment'] == 3) & (df_guesses['promote_type_seen'] == promote_type) & df_guesses['guesser_is_female'] == female]
    X = pd.DataFrame(
        sm.add_constant(
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
        ),
        columns = [
            'const', 'Self-evaluation', 'Female', 'Self-evaluation x Female'
        ] + [f"fe{i}" for i in range(10)],
        index = data.index
    )
    fitted2 = sm.OLS(data['wage_guess'], X).fit(
        cov_type='cluster', cov_kwds={'groups': data.index}
    )

    return make_table(
        {
            'Self-evaluation + gender': fitted1,
            'Self-evaluation + gender + performance*': fitted2
        },
        title = f'Wage guesses, with {"first" if promote_type == 1 else "second"} self-evaluation type and {"female" if female == 1 else "male"} guessers only',
    )

# %%
print(hyp5_6_table(1, 1))

# %%
print(hyp5_6_table(1, 2))

# %%
print(hyp5_6_table(0, 1))

# %%
print(hyp5_6_table(0, 2))


# %%
print('\n(*) Includes fixed effects on performance')
print('\nStandard errors robust to clustering')


