### AI works cited
generated variable figure in 0.2 by providing model intuition and variable dependencies
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



## Latent Variable Identification
The latent variable for this model is the team's momentum at a given point in time. It is not directly recorded anywhere in the dataset, but the data implies the existence of some type of momentum which contributes to the strength, rest differences, and win percentage, which are variables that covary together and with the target outcome. Teams in good form are more likely to handle fatigue, have strong short-term performance, and ultimately win more games by accumulating more momentum than opponents over the course of a season. It is natural to model this as a latent variable since it evolves over time and influences multiple observed variables. This latent variable corresponds directly to the role played by upstream nodes in the Bayes net, particularly the underlying causes behind HomeStrength, AwayStrength, and the rest-related variables. In the BN, these variables were treated as distinct parents of HomeWin, but they stood in for deeper, unmeasured causes of performance, which this HMM exogenizes by introducing a hidden state that causally generates these observed features and the outcome variable.
