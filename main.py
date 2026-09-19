# main.py — Corporate Heist final
# Strategy: CatBoost regression on train.csv, augmented with 150 dev winners
# weighted 5x, then ranked + dedupe + top 500 for submission.
import re
from collections import Counter
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

SEED = 42
ID = 'candidate_id'
TARGET = 'post_hire_score'

train = pd.read_csv('train.csv'); dev = pd.read_csv('dev.csv')
test  = pd.read_csv('test.csv');  winners = pd.read_csv('dev_winners.csv')
win_ids = set(winners['candidate_id'].astype(str).str.strip())

def p_years(x):
    if pd.isna(x): return np.nan
    m = re.search(r'([\d.]+)', str(x)); return float(m.group(1)) if m else np.nan
def p_tech(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if '/' in s:
        a,b = s.split('/',1)
        try: return float(re.findall(r'[\d.]+',a)[0])/float(re.findall(r'[\d.]+',b)[0])*100
        except: return np.nan
    m = re.search(r'([\d.]+)', s)
    if not m: return np.nan
    v = float(m.group(1)); return v*100 if v<=1 else v
def p_apt(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if '/' in s:
        a,b = s.split('/',1)
        try: return float(re.findall(r'[\d.]+',a)[0])/float(re.findall(r'[\d.]+',b)[0])*10
        except: return np.nan
    m = re.search(r'([\d.]+)', s); return float(m.group(1)) if m else np.nan
def p_rating(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if '/' in s:
        a,b = s.split('/',1)
        try: return float(re.findall(r'[\d.]+',a)[0])/float(re.findall(r'[\d.]+',b)[0])*10
        except: return np.nan
    m = re.search(r'([\d.]+)', s)
    if m: return float(m.group(1))
    for k,v in [('outstanding',10),('excellent',9),('good',7),('average',5),('poor',3)]:
        if k in s: return v
    return np.nan
def p_ctc(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().replace(',', '')
    m = re.search(r'([\d.]+)', s)
    if not m: return np.nan
    v = float(m.group(1))
    if 'cr' in s: v *= 1e7
    elif 'lpa' in s or 'lakh' in s or 'lac' in s: v *= 1e5
    return v
def p_notice(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if 'immediate' in s or 'available now' in s: return 0.0
    m = re.search(r'(\d+)', s)
    if not m: return np.nan
    d = float(m.group(1)); return d*30 if 'month' in s else d
def p_kpi(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ('y','yes','1','true'): return 1
    if s in ('n','no','0','false'): return 0
    return np.nan
def p_code(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().strip()
    if 'not tracked' in s or 'unknown' in s or s == '': return np.nan
    m = re.search(r'([\d.]+)', s); return float(m.group(1)) if m else np.nan

IIT = ['delhi','bombay','madras','kanpur','kharagpur','roorkee','guwahati','hyderabad',
       'banaras','bhu','varanasi','indore','patna','mandi','jodhpur','gandhinagar',
       'bhubaneswar','tirupati','palakkad','dhanbad','ism']
NIT = ['warangal','surathkal','karnataka','trichy','tiruchirappalli','calicut','rourkela',
       'durgapur','allahabad','jaipur','kurukshetra','silchar','hamirpur','patna','raipur',
       'agartala','nagpur','goa','meghalaya','mizoram','manipur','sikkim','arunachal',
       'jamshedpur','delhi','uttarakhand','puducherry']
def norm_inst(x):
    if pd.isna(x): return 'unknown'
    s = str(x).lower()
    if 'iit' in s or 'indian institute of technology' in s:
        for c in IIT:
            if c in s: return f'IIT {c.title()}'
        return 'IIT Other'
    if 'nit' in s or 'national institute of technology' in s:
        for c in NIT:
            if c in s: return f'NIT {c.title()}'
        return 'NIT Other'
    if 'iiit' in s: return 'IIIT'
    if 'bits' in s: return 'BITS'
    return str(x).strip()
def norm_ch(x):
    if pd.isna(x): return 'unknown'
    s = str(x).lower()
    if 'referral' in s: return 'Referral'
    if 'linkedin' in s: return 'LinkedIn'
    if 'campus' in s: return 'Campus'
    if 'portal' in s: return 'Job Portal'
    if 'direct' in s: return 'Direct'
    if 'sourcing' in s: return 'Sourcing'
    if 'walk' in s: return 'Walk-in'
    return 'Other'
def role_t(x):
    if pd.isna(x): return 1
    s = str(x).lower()
    if any(k in s for k in ['data scientist','data engineer','devops','ml engineer']): return 2
    if any(k in s for k in ['qa','product analyst','backend']): return 0
    return 1
def comp_t(x):
    if pd.isna(x): return 0
    s = str(x).strip()
    if s == '10000+': return 3
    if s in ('5000-9999','1000-4999'): return 2
    if s in ('500-999','100-500'): return 1
    return 0

def build(df):
    d = df.copy()
    d['exp_years'] = d['total_experience'].apply(p_years)
    d['tech'] = d['technical_assessment'].apply(p_tech)
    d['apt'] = d['aptitude_score'].apply(p_apt)
    d['rating'] = d['last_rating'].apply(p_rating)
    d['ctc'] = d['current_ctc'].apply(p_ctc)
    d['exp_ctc'] = d['expected_ctc'].apply(p_ctc)
    d['notice_d'] = d['notice_period'].apply(p_notice)
    d['kpi'] = d['kpi_met'].apply(p_kpi)
    d['code'] = d['public_code_contributions'].apply(p_code) if 'public_code_contributions' in d.columns else np.nan
    d['ctc_gap'] = (d['exp_ctc'] - d['ctc']) / (d['ctc'].abs() + 1)
    d['age_at_grad'] = d['age'] - (2026 - d['graduation_year'])
    d['exp_per_emp'] = d['exp_years'] / (d['num_employers'] + 1)
    d['skills_n'] = d['skills'].fillna('').astype(str).apply(lambda x: len([s for s in re.split(r'[|,;]', x) if s.strip()]))
    d['career_steps'] = d['career_path'].fillna('').astype(str).apply(lambda x: x.count('>') + x.count('→'))
    d['institute_n'] = d['institute'].apply(norm_inst)
    d['channel_n'] = d['recruitment_channel'].apply(norm_ch)
    d['role_tier'] = d['applied_role'].apply(role_t)
    d['company_tier'] = d['company_size'].apply(comp_t)
    d['is_referral'] = (d['channel_n'] == 'Referral').astype(int)
    d['is_portal'] = (d['channel_n'] == 'Job Portal').astype(int)
    d['is_recent_grad'] = (d['graduation_year'] >= 2017).astype(int)
    d['tech_x_exp'] = d['tech'] * d['exp_years'].fillna(0)
    d['tech_x_role'] = d['tech'] * d['role_tier']
    d['tech_over_apt'] = d['tech'] / (d['apt'].fillna(5) + 1)
    d['exp_sq'] = d['exp_years'] ** 2
    d['age_minus_exp'] = d['age'] - d['exp_years']
    return d

train_f = build(train); dev_f = build(dev); test_f = build(test)

# skills tokens
def tk(s): return [t.strip().lower() for t in re.split(r'[|,;]', str(s)) if t.strip()] if pd.notna(s) else []
all_sk = []
for s in train_f['skills'].fillna(''): all_sk.extend(tk(s))
top_sk = [s for s,_ in Counter(all_sk).most_common(40)]
skill_cols = []
for sk in top_sk:
    col = 'sk_' + re.sub(r'[^a-z0-9]', '_', sk)[:22]
    if col in train_f.columns: continue
    train_f[col] = train_f['skills'].apply(lambda x: 1 if sk in tk(x) else 0)
    dev_f[col]   = dev_f['skills'].apply(lambda x: 1 if sk in tk(x) else 0)
    test_f[col]  = test_f['skills'].apply(lambda x: 1 if sk in tk(x) else 0)
    skill_cols.append(col)

num_cols = ['age','graduation_year','num_employers','trainings_last_year','training_hours',
            'exp_years','tech','apt','rating','ctc','exp_ctc','notice_d','kpi',
            'ctc_gap','age_at_grad','exp_per_emp','skills_n','career_steps',
            'role_tier','company_tier','is_referral','is_portal','is_recent_grad',
            'tech_x_exp','tech_x_role','tech_over_apt','exp_sq','age_minus_exp'] + skill_cols
cat_cols = ['institute_n','degree','major','current_city','applied_role','company_type',
            'company_size','channel_n','last_job_change','currently_enrolled',
            'overtime_history','awards']
feats = num_cols + cat_cols

def prep(df):
    X = df[feats].copy()
    for c in cat_cols: X[c] = X[c].fillna('missing').astype(str)
    for c in num_cols: X[c] = X[c].fillna(train_f[c].median())
    return X

X_tr = prep(train_f); X_dv = prep(dev_f); X_te = prep(test_f)
y_tr = train_f[TARGET].values

# augment with 150 winners at weight 5
HIGH = np.quantile(y_tr, 0.95)
dev_f['is_winner'] = dev_f[ID].astype(str).str.strip().isin(win_ids).astype(int)
win_rows = dev_f[dev_f['is_winner'] == 1].copy()
win_rows[TARGET] = HIGH

X_aug = pd.concat([X_tr, prep(win_rows)], ignore_index=True)
y_aug = np.concatenate([y_tr, np.full(len(win_rows), HIGH)])
w_aug = np.concatenate([np.ones(len(X_tr)), np.full(len(win_rows), 5.0)])

# ensemble 3 seeds (rank-averaged)
dev_preds, te_preds = [], []
for seed in (42, 7, 2024):
    m = CatBoostRegressor(iterations=900, learning_rate=0.05, depth=7,
                          verbose=0, random_seed=seed, l2_leaf_reg=4)
    m.fit(X_aug, y_aug, sample_weight=w_aug, cat_features=cat_cols)
    dev_preds.append(m.predict(X_dv))
    te_preds.append(m.predict(X_te))

for i, p in enumerate(dev_preds): test_f[f'rp{i}'] = pd.Series(te_preds[i]).rank(pct=True).values
test_f['base'] = test_f[[f'rp{i}' for i in range(3)]].mean(axis=1)

# code contributions boost (test only)
def cb(x):
    if pd.isna(x): return 0.0
    x = float(x)
    if x <= 0: return 0.0
    if x <= 5: return 0.005
    if x <= 20: return 0.015
    if x <= 50: return 0.030
    return 0.050
test_f['code_boost'] = test_f['code'].apply(cb)
test_f['final'] = test_f['base'] + test_f['code_boost']

# dedupe by email/phone
def dedupe(df, col):
    d = df.copy()
    d['_e'] = d['email'].fillna('').astype(str).str.lower().str.strip()
    d['_p'] = d['phone'].fillna('').astype(str).str.replace(r'\D','',regex=True)
    d = d.sort_values(col, ascending=False).reset_index(drop=True)
    se, sp, keep = set(), set(), []
    for i, r in d.iterrows():
        if (r['_e'] and r['_e'] in se) or (r['_p'] and r['_p'] in sp): continue
        if r['_e']: se.add(r['_e'])
        if r['_p']: sp.add(r['_p'])
        keep.append(i)
    return d.loc[keep].reset_index(drop=True)

final = dedupe(test_f, 'final').head(500)
sub = final[[ID]].copy()
sub['rank'] = range(1, len(sub) + 1)
sub = sub[['rank', ID]]
sub.to_csv('submission.csv', index=False)

# sanity
assert len(sub) == 500
assert list(sub['rank']) == list(range(1, 501))
assert len(set(sub[ID])) == 500
print("✅ submission.csv written:", len(sub), "rows")