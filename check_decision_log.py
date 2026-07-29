import pandas as pd

df = pd.read_csv('data/output_data/decision_logs/mplight_TRAIN_FULL_tsc_rl_adversarial_rank0_eps50.csv')
print(f"Total rows: {len(df)}")
print(df)
# print(f"Episodes covered: {df['episode'].min()} to {df['episode'].max()}")
#
# # How does the attacker / controller behavior evolve?
# summary = df.groupby('episode').agg(
#     fakes_mean=('n_fakes_injected', 'mean'),
#     most_used_phase_pct=('chosen_phase', lambda x: x.value_counts().iloc[0]/len(x)*100),
#     phases_used=('chosen_phase', 'nunique'),
# ).round(2)
# print(summary.iloc[::20])  # every 20th episode
