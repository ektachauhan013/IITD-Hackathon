import pandas as pd

dev = pd.read_csv('dev.csv')
w   = pd.read_csv('dev_winners.csv')

w['candidate_id']   = w['candidate_id'].astype(str).str.strip()
dev['candidate_id'] = dev['candidate_id'].astype(str).str.strip()

win_ids = set(w['candidate_id'])
dev['is_winner'] = dev['candidate_id'].isin(win_ids)

print("Winners in dev.csv:", int(dev['is_winner'].sum()), "/ 150")

# Numeric comparison
num_cols = dev.select_dtypes(include='number').columns
for c in num_cols:
    print(f"\n--- {c} ---")
    print(dev.groupby('is_winner')[c].agg(['mean', 'median', 'count']))

# Text-level clues
print("\n=== Applied role distribution ===")
print(pd.crosstab(dev['applied_role'], dev['is_winner'], normalize='columns').round(3))

print("\n=== Recruitment channel distribution ===")
print(pd.crosstab(dev['recruitment_channel'], dev['is_winner'], normalize='columns').round(3))

print("\n=== Company size distribution ===")
print(pd.crosstab(dev['company_size'].fillna('missing'), dev['is_winner'], normalize='columns').round(3))

print("\n=== Notice period raw values (winners) ===")
print(dev.loc[dev['is_winner'], 'notice_period'].value_counts().head(15))

print("\n=== Notice period raw values (non-winners) ===")
print(dev.loc[~dev['is_winner'], 'notice_period'].value_counts().head(15))

print("\n=== Top 15 institutes among winners ===")
print(dev.loc[dev['is_winner'], 'institute'].value_counts().head(15))

print("\n=== Top 15 institutes among non-winners ===")
print(dev.loc[~dev['is_winner'], 'institute'].value_counts().head(15))