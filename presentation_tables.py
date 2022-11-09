# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re

# %%
df_bids = pd.read_csv('employer_wage_bids.csv', index_col='employer')
df_bids['bid'] = df_bids['bid'] * 100
df_guesses = pd.read_csv('applicant_wage_guesses.csv', index_col='guesser')
df_guesses['wage_guess'] = df_guesses['wage_guess'] * 100
df_app = pd.read_csv('applicant_data_clean.csv', index_col='applicant')

# %%
def signif_level(pvalue):
    if pvalue < 0.01:
        return "***"
    elif pvalue < 0.05:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return ""

def make_table(fitted_models, title=None, notes=None, colwidth=8):
    tables = []
    for name, fitted in fitted_models.items():
        tables.append(
            pd.DataFrame(
                [
                    rf"\shortstack{{{param:.3g} ({bse:.2g}){signif_level(p)}}}"
                    for param, bse, p in zip(fitted.params, fitted.bse, fitted.pvalues)
                ],
                index=fitted.params.index,
                columns = [name.replace('+', r'\newline +').replace('*', r'$^\dagger$')],
            )
        )
    df = pd.concat(tables, axis=1)
    df = df.loc[[i for i in df.index if not i.startswith('fe')]]
    sty = df.style.format(
        na_rep = ''
    ).format_index(
        escape="latex", axis=0
    )
    
    ncols = len(fitted_models)
    tab = sty.to_latex(
        column_format = 'l' + f'p{{{colwidth}em}}'*ncols,
        hrules = True,
        caption = title,
        position_float='centering',
    )

    if notes is not None:
        maxlen = max(len(note) for note in notes)
        tab = re.sub(
            r'(?=\n\\end{tabular})',
            "\n" + rf'\\multicolumn{{{ncols}}}{{p{{{max(maxlen + 6, colwidth*ncols)}ex}}}}{{\\textit{{Notes}}: ' + r' \\newline\\quad '.join(notes) + '}', tab)
    return tab

# %%
def hyp1_3_table(promote_type = 1):
    data = df_bids[
        (df_bids['treatment'] == 1) & (df_bids['promote_type_seen'] == promote_type)
    ]
    X = sm.add_constant(
        data[f'app_promote{promote_type}']
    )
    X.columns=['const', 'Self-evaluation']

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
            r'Self-evaluation + gender': fitted2,
            r'Self-evaluation + gender + performance*': fitted3
        },
        title = f'Employer bids, with {"first" if promote_type == 1 else "second"} self-evaluation type',
        notes = ['*$p<0.1$, **$p<0.05$, ***$p<0.01$.', 'Standard errors clustered by employer.', '(†) indicates inclusion of performance fixed effects.']
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
        notes = ['*$p<0.1$, **$p<0.05$, ***$p<0.01$.', 'Standard errors clustered by applicant.', '(†) indicates inclusion of performance fixed effects.'],
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
            r'Second \newline self-evaluation type': hyp4_fit(2),
        },
        title = 'Wage guesses (self-evaluation-only treatment)',
        notes = ['*$p<0.1$, **$p<0.05$, ***$p<0.01$.', 'Standard errors clustered by guesser.',],
        colwidth = 12,
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
        notes = ['*$p<0.1$, **$p<0.05$, ***$p<0.01$.', 'Standard errors clustered by guesser.', '(†) indicates inclusion of performance fixed effects.'],
    )

# %%
print(hyp5_6_table(1, 1))

# %%
print(hyp5_6_table(1, 2))

# %%
print(hyp5_6_table(0, 1))

# %%
print(hyp5_6_table(0, 2))



