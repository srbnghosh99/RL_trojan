import pandas as pd
r = pd.read_csv("runs.csv")
# 1) keep only finished runs
r = r[r.n_test_ep >= 150]
# 2) drop exact-duplicate results (same agent/task and identical final value)
r = r.drop_duplicates(subset=["agent","task","final_tt_lastN"])
# 3) inspect what survived before trusting any average
print(r[["file","agent","task","n_test_ep","final_tt_lastN"]].to_string(index=False))
r.to_csv("runs_clean.csv", index=False)
