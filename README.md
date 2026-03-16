# Final Project: NBA Win Prediction

Dataset: https://www.kaggle.com/datasets/nathanlauga/nba-games

## Motivation, Problem Statement & PEAS

Predicting NBA game outcomes is a key problem in sports analytics with applications to strategies, media, and sportsbetting. Each game involves teams whose relative quality, scheduling fatigue, and recent performance all contribute to determining the winner. Uncertainty modeling is necessary because no deterministic method can model perfectly the randomness of NBA games; even a heavily favored team loses roughly 20–30% of the time. Win probability is thus more useful than classification because we are more interested in probabilities than labeling predictions. Observable features such as rest days and season win-percentage are noisy variables that function similarly to team strength, and a probabilistic model propagates that noise through inference. Non-probabilistic approaches are limited, for example decision trees cut continuous features and cannot express a statement like "70% home win" Logistic regression similarly provides no explicit causal graph. These issues point us to a probabilistic approach.

### PEAS

The performance measure is accuracy, F1-score, and log-likelihood, with higher accuracy and F1 indicating better predictions (as in the last model) and higher log-likelihood indicating better  probability estimates. The environment is the set of NBA games from the past several years, which is partially observable because true team form is a latent variable that we can only approximate from prior results. The actuator is the probability the model outputs: P(HomeWin = 1 | evidence). The sensors are the observable pre-game features like home and away team overall win %, number of rest days since the team's last game, and the rolling 10-game win% variable we add on to the first model.

## Dataset & Preprocessing

The dataset comes from Kaggle, and is published by nathanlauga. games.csv contains 26,825 regular-season games spanning the 2003-2022 seasons. Another file ranking.csv provides daily standings snapshots with team win percentages and provides data for form features.

The task is binary classification given only features observable prior to the game, no in-game statistics, to predict whether the home team wins or loses.

HomeWin is a binary target variable. HomeWinPct and AwayWinPct are continuous season win-percentage values that are discretized into the ordinal categories Weak, Average, and Strong. HomeRestDays and AwayRestDays are integer counts that are discretized into BackToBack, NormalRest, and WellRested. WinPctDiff is the difference between home and away win percentages, binned into AwayStronger, Even, and HomeStronger. RestDiff is the difference in rest days, binned into AwayRestAdv, EvenRest, and HomeRestAdv. HomeRoll10 and AwayRoll10 are new continuous features, binned into Cold, Average, and Hot.

### Preprocessing Steps

Season win percentage is joined to each game using pandas merge_asof in the backward direction, so only rankings published before the game date are used, preventing future data from leaking into the features. Rest days are computed by iterating over games sorted by date and showing the gap since each team's most recent appearance. The rolling win percentage is computed by building a flat per team game result log that combines home and away appearances in chronological order, then applying shift(1).rolling(10).mean(). All continuous features are discretized using fixed thresholds based on domain knowledge. Rows containing NaN in the feature set is from mostly early games and are dropped since they make up a small portion of the data. The test/train split is made on HomeWin.

### Justifications for design

Temporal ordering is enforced throughout so that no information from the future enters the model's inputs. The GAME_ID join for rolling statistics is a deliberate departure from the original merge_asof approach: because shift(1).rolling() is applied inside the sorted log before the join, the result for any given game reflects only that team's performance in JUST prior games. The discretization reduces sparsity in the CPTs and keeps the model interpretable.
```python
# Preprocessing
games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
ranking['STANDINGSDATE'] = pd.to_datetime(ranking['STANDINGSDATE'])
games = games.sort_values('GAME_DATE_EST').reset_index(drop=True)
ranking = ranking.sort_values('STANDINGSDATE').reset_index(drop=True)

def attach_win_pct(games_df, ranking_df, team_col, out_col):
    side = games_df[['GAME_ID', team_col, 'GAME_DATE_EST']].copy().sort_values('GAME_DATE_EST')
    side = pd.merge_asof(
        side,
        ranking_df[['TEAM_ID', 'STANDINGSDATE', 'W_PCT']].sort_values('STANDINGSDATE'),
        left_on='GAME_DATE_EST', right_on='STANDINGSDATE',
        left_by=team_col,        right_by='TEAM_ID',
        direction='backward'
    )
    return side[['GAME_ID', 'W_PCT']].rename(columns={'W_PCT': out_col})

games = games.merge(attach_win_pct(games, ranking, 'HOME_TEAM_ID',    'HomeWinPct'), on='GAME_ID', how='left')
games = games.merge(attach_win_pct(games, ranking, 'VISITOR_TEAM_ID', 'AwayWinPct'), on='GAME_ID', how='left')

last_game = {}
home_rest = []
away_rest = []
for _, row in games.iterrows():
    d, h, a = row['GAME_DATE_EST'], row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']
    home_rest.append((d - last_game[h]).days if h in last_game else 2)
    away_rest.append((d - last_game[a]).days if a in last_game else 2)
    last_game[h] = d
    last_game[a] = d

games['HomeRestDays'] = home_rest
games['AwayRestDays'] = away_rest
```
```python
# Rolling 10-game win % function
ROLL_WINDOW = 10

home_log = games[['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID', 'HOME_TEAM_WINS']].copy()
home_log.rename(columns={'HOME_TEAM_ID': 'team_id', 'HOME_TEAM_WINS': 'win'}, inplace=True)

away_log = games[['GAME_DATE_EST', 'GAME_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']].copy()
away_log['win'] = 1 - away_log['HOME_TEAM_WINS']
away_log.rename(columns={'VISITOR_TEAM_ID': 'team_id'}, inplace=True)
away_log = away_log[['GAME_DATE_EST', 'GAME_ID', 'team_id', 'win']]

team_log = (
    pd.concat(
        [home_log[['GAME_DATE_EST', 'GAME_ID', 'team_id', 'win']], away_log],
        ignore_index=True
    )
    .sort_values(['team_id', 'GAME_DATE_EST', 'GAME_ID'])
    .reset_index(drop=True)
)

team_log['roll_wpct'] = (
    team_log
    .groupby('team_id', group_keys=False)['win']
    .apply(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=3).mean())
)

home_roll = (
    team_log
    .merge(games[['GAME_ID', 'HOME_TEAM_ID']], on='GAME_ID', how='inner')
    .query('team_id == HOME_TEAM_ID')
    [['GAME_ID', 'roll_wpct']]
    .rename(columns={'roll_wpct': 'HomeRoll10'})
)

away_roll = (
    team_log
    .merge(games[['GAME_ID', 'VISITOR_TEAM_ID']], on='GAME_ID', how='inner')
    .query('team_id == VISITOR_TEAM_ID')
    [['GAME_ID', 'roll_wpct']]
    .rename(columns={'roll_wpct': 'AwayRoll10'})
)

games = games.merge(home_roll, on='GAME_ID', how='left')
games = games.merge(away_roll, on='GAME_ID', how='left')

n_nan_home = games['HomeRoll10'].isna().sum()
n_nan_away = games['AwayRoll10'].isna().sum()
print(f'HomeRoll10 NaNs (early-season games, will bin to Average): {n_nan_home}/{len(games)}')
print(f'AwayRoll10 NaNs: {n_nan_away}/{len(games)}')
print()
print(games[['GAME_DATE_EST','HOME_TEAM_ID','VISITOR_TEAM_ID','HomeRoll10','AwayRoll10']].head(5))
```
```python
# Discretization
def bin_strength(p):
    if pd.isna(p): return np.nan
    if p <= 0.40:  return 'Weak'
    if p <= 0.60:  return 'Average'
    return 'Strong'

def bin_rest(days):
    if days == 0:  return 'BackToBack'
    if days <= 2:  return 'NormalRest'
    return 'WellRested'

def bin_roll(p):
    if pd.isna(p): return 'Average'
    if p <= 0.35:  return 'Cold'
    if p <= 0.65:  return 'Average'
    return 'Hot'

games['HomeStrength'] = games['HomeWinPct'].apply(bin_strength)
games['AwayStrength'] = games['AwayWinPct'].apply(bin_strength)
games['HomeRest_bin'] = games['HomeRestDays'].apply(bin_rest)
games['AwayRest_bin'] = games['AwayRestDays'].apply(bin_rest)
games['HomeRoll_bin'] = games['HomeRoll10'].apply(bin_roll)
games['AwayRoll_bin'] = games['AwayRoll10'].apply(bin_roll)

games['WinPctDiff'] = games['HomeWinPct'] - games['AwayWinPct']
games['WinPctDiff_bin'] = pd.cut(games['WinPctDiff'], bins=[-np.inf, -0.10, 0.10, np.inf], labels=['AwayStronger', 'Even', 'HomeStronger'])
games['RestDiff'] = games['HomeRestDays'] - games['AwayRestDays']
games['RestDiff_bin'] = pd.cut(games['RestDiff'], bins=[-np.inf, -1, 0, np.inf], labels=['AwayRestAdv', 'EvenRest', 'HomeRestAdv'])
games['HomeWin'] = games['HOME_TEAM_WINS'].astype(int)

print('Discretization complete.')
print(games[['HomeStrength','AwayStrength','WinPctDiff_bin', 'HomeRest_bin','AwayRest_bin','RestDiff_bin','HomeRoll_bin','AwayRoll_bin','HomeWin']].head(3))
```
## Model Choice & Methodology

### Model Selection

This project uses a Bayesian Network implemented with the pgmpy library, as in the previous model. A BN is well suited to this problem because it encodes a causal story with a DAG that follows intuition of domain knowledge (stronger teams beat weaker ones, rested teams have an edge, and recent form reflects short-term momentum). The graphical structure also supports partial evidence if rest data is unavailable for a particular game; variable elimination marginalises it and returns a valid posterior. The model naturally expresses uncertainty through CPTs.

### Model Improvements

The baseline model uses season-long win percentage and rest differential as its only predictors, estimated with MLE. The improved model adds two new nodes that represent each team's win rate over their most recent 10 games. These nodes connect directly to HomeWin in the DAG, giving the model an independent signal about short-term momentum that the season-long win percentage cannot capture. Both models use the same MLE CPT estimation procedure, the same Variable Elimination inference engine, and the same evidence columns at prediction time. The key structural change is that HomeWin in this model has 2 more parents, making its CPT wider.

### Model Structure

The DAG is shown in the diagram below. HomeStrength and AwayStrength each point to WinPctDiff_bin (because the difference between team qualities determines which direction the win-percentage gap falls) and both also point directly to HomeWin. HomeRest_bin and AwayRest_bin each point to RestDiff_bin, which in turn points to HomeWin. The two new nodes HomeRoll_bin and AwayRoll_bin each point directly to HomeWin. This structure was validated by chi-square conditional independence tests: in every case the p-value was well below 0.05, confirming that none of the parent-child edges in the graph should be removed. CPT's are estimated by MLE. Variable Elimination computes P(HomeWin | E) according to the joint distribution defined by the CPTs. The MAP prediction is obtained by choosing whichever value of HomeWin has the higher posterior probability, which corresponds to the threshold of 0.5.

## Training and Implementation

## Results and Conclusion

## Citations
● pgmy
● External resources:
● ChatGPT: implementing rolling 10-game win function, conditional chi-squared test logic

