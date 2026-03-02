### 0.1 PEAS / Agent Analysis
The goal is to predict NBA game outcomes over time by using a hidden team 
condition that influences observable variables. Equivalently, we want to learn 
how hidden states transition from one game to the next and how those states 
generate observable features and results. Probabilistic modeling is suited to this
problem because there is uncertainty in the way the data is generated and how
performance is driven. 
### 0.2 Agent Setup, Data Preprocessing, Training Setup
The data includes game outcomes, team data, dates, and season-level performance data. This structure allows for analysis at both the individual game level and sequential level when ordered by date. 
Variables:

GAME_ID: A unique identifier for each game.

GAME_DATE_EST: The calendar date on which the game was played. This variable is used for ordering games in time, computing rest days between games, and restructuring the data into sequences for HMM training.

HOME_TEAM_ID/VISITOR_TEAM_ID: Identifiers for the home and away teams. These variables define the matchup and allow games to be grouped by team to form temporal sequences.

HOME_TEAM_WINS: A binary indicator of whether the home team won the game. This is the outcome of interest and represents the agent’s prediction target.

HomeStrength/AwayStrength: indicator of the home/away team’s overall strength entering the game, derived from season-level statistics such as win percentage. Represents long-run team quality.

WinPctDiff_bin: Binned difference between home and away win percentages. Summarizes relative team strength.

HomeRest_bin/AwayRest_bin: Binned number of days the home/away team rested before the game. Captures short-term fatigue and recovery effects.

RestDiff_bin
Binned difference between home and away rest days. Represents scheduling advantage/disadvantage.

Season-level team statistics such as W, L, W_PCT: These summarize team performance up to a point in the season. They serve as measures for team quality/strength and are explanatory variables for game outcomes.

Conference: categorical variable indicating whether a team belongs to the Eastern or Western Conference. This provides information about competition level and scheduling patterns but is secondary compared to direct performance measures.

Game outcomes are primarily driven by team quality and home/away context. Variables such as wins, losses, and strength of schedule summarize a team’s historical performance and influence the probability of winning an upcoming game. Since these statistics are cumulative, observations across games are temporally dependent where each game updates their commulative performance, affecting future outcomes.

To capture this dependency, the model uses an HMM. The hidden state represents an unobserved team condition--which we will call "momentum"-- that evolves from one game to the next. This latent state generates the observed variables, including game outcomes and season-level performance summaries. This model structure is appropriate, because it explicitly models dependence over time, and allows the observed variables to be driven by a single latent factor, which is more realistic than treating each game as independent.
```markdown
                     Hidden State
                  (form / momentum)
                          |
                          v
        ------------------------------------------------
        |                 |                 |         |
        v                 v                 v         v
   Team Strength      Rest Context      Relative        Outcome
 (HomeStrength,    (HomeRest_bin,     Differences     (HOME_TEAM_WINS)
  AwayStrength)     AwayRest_bin)   (WinPctDiff_bin,
                                      RestDiff_bin)
```

## Latent Variable Identification
The latent variable for this model is the team's momentum at a given point in time. It is not directly recorded anywhere in the dataset, but the data implies the existence of some type of momentum which contributes to the strength, rest differences, and win percentage, which are variables that covary together and with the target outcome. Teams in good form are more likely to handle fatigue, have strong short-term performance, and ultimately win more games by accumulating more momentum than opponents over the course of a season. It is natural to model this as a latent variable since it evolves over time and influences multiple observed variables. This latent variable corresponds directly to the role played by upstream nodes in the Bayes net, particularly the underlying causes behind HomeStrength, AwayStrength, and the rest-related variables. In the BN, these variables were treated as distinct parents of HomeWin, but they stood in for deeper, unmeasured causes of performance, which this HMM exogenizes by introducing a hidden state that causally generates these observed features and the outcome variable.

## Temporal Data Construction
For this HMM, the data is converted into a sequence through natural temporal ordering. This is done by grouping games by team and sorting them by GAME_DATE_EST, so each sequence represents the order of games played by a team over time. Each time step is simply the next game that team plays, which reflects the real flow of the season. Since the HMM is discrete, continuous and high-cardinality variables are discretized into a small number of bins. Team strength is binned into low, medium, and high using win percentage quantiles, while rest variables are grouped into short, normal, and long rest based on days off between games. Difference variables are binned into negative, neutral, and positive categories. These choices keep the model simple, reduce overfitting, and preserve the main patterns in the data. 
```python
ranking = ranking.sort_values("STANDINGSDATE")
games = games.sort_values("GAME_DATE_EST")

home_strength = pd.merge_asof(
    games,
    ranking,
    left_on="GAME_DATE_EST",
    right_on="STANDINGSDATE",
    left_by="HOME_TEAM_ID",
    right_by="TEAM_ID",
    direction="backward"
)

away_strength = pd.merge_asof(
    games,
    ranking,
    left_on="GAME_DATE_EST",
    right_on="STANDINGSDATE",
    left_by="VISITOR_TEAM_ID",
    right_by="TEAM_ID",
    direction="backward",
    suffixes=("", "_away")
)

games = home_strength.merge(
    away_strength[["GAME_ID", "W_PCT"]],
    on="GAME_ID",
    suffixes=("_home", "_away")
)
```
^ ensures each game only sees information available at the time using merge_asof

```python
# each sequence is a team's games ordered in time
def build_team_sequences(df):
    sequences = {}

    for team_id in pd.unique(
        df[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]].values.ravel()
    ):
        team_games = df[
            (df["HOME_TEAM_ID"] == team_id) |
            (df["VISITOR_TEAM_ID"] == team_id)
        ].sort_values("GAME_DATE_EST")

        obs = team_games[
            [
                "HomeStrength",
                "AwayStrength",
                "HomeRest_bin",
                "AwayRest_bin",
                "WinPctDiff_bin",
                "RestDiff_bin",
                "HOME_TEAM_WINS"
            ]
        ].dropna()

        if len(obs) > 1:
            sequences[team_id] = obs.values.tolist()

    return sequences
team_sequences = build_team_sequences(games)
```
## HMM Implementation with EM
### Model Design
Z_t ∈ {LowForm, MediumForm, HighForm}
Number of states = 3

We use a single composite discrete observation symbol constructed from discretized variables: (HomeStrength, AwayStrength, HomeRest_bin, AwayRest_bin, WinPctDiff_bin, RestDiff_bin, HomeWin)

```python
def encode_observations(df, obs_cols):
    obs_tuples = df[obs_cols].astype(str).agg("_".join, axis=1)
    symbols, encoded = np.unique(obs_tuples, return_inverse=True)
    return encoded, symbols
```
```python
def initialize_hmm(K, M, seed=0):
    rng = np.random.default_rng(seed)

    pi = rng.random(K)
    pi /= pi.sum()

    A = rng.random((K, K))
    A /= A.sum(axis=1, keepdims=True)

    B = rng.random((K, M))
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B
```
E-Step: 
```python
def forward_backward(obs, pi, A, B):
    T = len(obs)
    K = len(pi)

    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))

    alpha[0] = pi * B[:, obs[0]]
    alpha[0] /= alpha[0].sum()

    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = B[j, obs[t]] * np.sum(alpha[t-1] * A[:, j])
        alpha[t] /= alpha[t].sum()

    beta[-1] = 1
    for t in reversed(range(T - 1)):
        for i in range(K):
            beta[t, i] = np.sum(A[i] * B[:, obs[t+1]] * beta[t+1])
        beta[t] /= beta[t].sum()

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        denom = np.sum(alpha[t][:, None] * A * B[:, obs[t+1]] * beta[t+1])
        for i in range(K):
            xi[t, i] = alpha[t, i] * A[i] * B[:, obs[t+1]] * beta[t+1] / denom

    return gamma, xi
```
M-Step:
```python
def m_step(obs, gamma, xi, K, M):
    pi = gamma[0]

    A = xi.sum(axis=0)
    A /= A.sum(axis=1, keepdims=True)

    B = np.zeros((K, M))
    for k in range(K):
        for t, o in enumerate(obs):
            B[k, o] += gamma[t, k]
        B[k] /= B[k].sum()

    return pi, A, B
```
Training:
```python
def train_hmm(obs, K, M, n_iters=20):
    pi, A, B = initialize_hmm(K, M)

    for _ in range(n_iters):
        gamma, xi = forward_backward(obs, pi, A, B)
        pi, A, B = m_step(obs, gamma, xi, K, M)

    return pi, A, B
team_id, seq = next(iter(team_sequences.items()))
seq_df = pd.DataFrame(seq, columns=obs_cols)
obs_encoded, obs_symbols = encode_observations(seq_df, obs_cols)
K = 3
M = len(obs_symbols)
pi_hat, A_hat, B_hat = train_hmm(obs_encoded, K, M)
```
### Connection to BN
The hidden state Z_t corresponds to the implicit causes in the BN that drive HomeStrength/AwayStrength, Rest variables, and HOME_TEAM_WINS. The HMM unifies these as a single latent variable that evolves over time, whereas the BN modeled them as separate parent nodes.

BN priors over team condition → π

BN temporal persistence intuition → A

BN CPTs of observed nodes → B
## Inference on Fitted HMM
The inference was performed with forward algorithm because through sequential filtering, we can predict the next game outcome using only past evidence. Viterbi uses the whole sequence, including future observations, to decode states, making it non-causal. For deciding before a game is played, only the filtered distribution alpha at t is valid. The Forward algorithm respects this causality constraint, so it is the inference method used for the prediction task.
```python
def forward_alg(observations, pi, A, B):
    T = len(observations)
    N = len(pi)

    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, observations[0]]
    alpha[0] /= alpha[0].sum()

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, observations[t]] * np.sum(
                alpha[t-1] * A[:, j]
            )
        alpha[t] /= alpha[t].sum()

    return alpha
```
```python
def predict_home_win_prob(alpha_prev, A, B, home_win_index):
    state_pred = alpha_prev @ A

    p_home_win = state_pred @ B[:, home_win_indices].sum(axis=1)

    return float(p_home_win)

def hmm_game_predictions(observations, pi, A, B, home_win_index):

    alpha = forward_alg(observations, pi, A, B)
    probs = []
    preds = []
    for t in range(1, len(observations)):
        p = predict_home_win_prob(alpha[t-1], A, B, home_win_index)
        probs.append(p)
        preds.append(int(p >= 0.5))

    return np.array(probs), np.array(preds)
```
```python
team_id, seq = next(iter(team_sequences.items()))
seq_df = pd.DataFrame(seq, columns=obs_cols)
obs_encoded, obs_symbols = encode_observations(seq_df, obs_cols)

# Identify which encoded symbols correspond to HOME_TEAM_WINS == 1
home_win_mask    = np.array([s.endswith('_1') for s in obs_symbols])
home_win_indices = np.where(home_win_mask)[0]

print(f"Total observation symbols : {len(obs_symbols)}")
print(f"Symbols encoding home win : {home_win_indices.sum()}")
```
```python
# Predictions for all teams
all_true, all_pred = [], []

for tid, seq in team_sequences.items():
    if len(seq) < 3:
        continue
    s_df   = pd.DataFrame(seq, columns=obs_cols)
    enc, _ = encode_observations(s_df, obs_cols)

    # Identify home-win indices for this team's local symbol set
    syms = _
    hw_idx = np.where(np.array([s.endswith('_1') for s in syms]))[0]
    if len(hw_idx) == 0:
        continue

    try:
        pi_t, A_t, B_t = train_hmm(enc, K=3, M=len(syms), n_iters=20)
        pr, pd_ = hmm_game_predictions(enc, pi_t, A_t, B_t, hw_idx)
        yt = s_df['HOME_TEAM_WINS'].astype(int).values[1:]
        all_true.extend(yt.tolist())
        all_pred.extend(pd_.tolist())
    except Exception:
        continue

all_true = np.array(all_true)
all_pred = np.array(all_pred)

print(f"Overall across {len(team_sequences)} teams")
print(f"  Total predictions : {len(all_true)}")
print(f"  Accuracy          : {np.mean(all_true == all_pred):.3f}")
print(f"  Baseline (always predict home win): {all_true.mean():.3f}")

```
Overall across 30 teams
  Total predictions : 41077
  Accuracy          : 0.511
  Baseline (always predict home win): 0.595

## Evaluation and Reflection
```python
def forward_loglik(observations, pi, A, B):
    T = len(observations)
    K = len(pi)
    alpha = np.zeros(K)
    log_lik = 0.0

    alpha = pi * B[:, observations[0]]
    c = alpha.sum()
    if c < 1e-300:
        return -np.inf
    alpha /= c
    log_lik += np.log(c)

    for t in range(1, T):
        alpha_new = np.array([
            B[j, observations[t]] * np.sum(alpha * A[:, j])
            for j in range(K)
        ])
        c = alpha_new.sum()
        if c < 1e-300:
            return -np.inf
        alpha = alpha_new / c
        log_lik += np.log(c)

    return log_lik


pi_init, A_init, B_init = initialize_hmm(K=3, M=len(obs_symbols), seed=0)
ll_before = forward_loglik(obs_encoded, pi_init, A_init, B_init)
ll_after  = forward_loglik(obs_encoded, pi_hat,  A_hat,  B_hat)

print(f"Log-likelihood before EM : {ll_before:.3f}")
print(f"Log-likelihood after  EM : {ll_after:.3f}")
print(f"Improvement              : {ll_after - ll_before:+.3f}  (positive = EM improved fit)")
print(f"Per-observation avg      : {ll_after / len(obs_encoded):.4f}  nats/game")

```
Log-likelihood before EM : -8716.371
Log-likelihood after  EM : -7590.372
Improvement              : +1125.999  (positive = EM improved fit)
Per-observation avg      : -4.9806  nats/game
### Reflection
What the HMM captures that the BN does not: the BN models each game in isolation and it has no memory of previous games, so a team on a five game winning streak gets the same prediction as a team that just lost five straight, if their observable features match. The HMM adds a hidden state that evolves across games, so recent history shapes the current belief about a team's momentum and, through it: the next prediction.
What assumptions the HMM imposes that the BN does not require: 3 assumptions come with this. First, form at game t depends on only form at game t−1, not on anything further back, if momentum actually builds over several weeks, a single previous state may not capture it. Second, once you know the hidden state, each game's observations are independent of all other games, and direct carry-over between observable variables across games is ignored unless it runs through the hidden state. Third, the number of states must be fixed before training, whereas the BN's structure came from domain knowledge and needed no such arbitrary choice.

### AI works cited
generated variable figure in 0.2 by providing model intuition and variable dependencies
created helper functions for deriving variables
created helper function for encoding observations, creating sequences
implemented the predictions for all teams given the forward algorithm function
