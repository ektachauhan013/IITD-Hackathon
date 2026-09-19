# cv_check.py — honest evaluation of dev-winner augmentation
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

# --- PARSERS ---
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

# --- FEATURES ---
num_cols = ['age','graduation_year','num_employers','trainings_last_year','training_hours',
            'exp_years','tech','apt','rating','ctc','exp_ctc','notice_d','kpi',
            'ctc_gap','age_at_grad','exp_per_emp','skills_n','career_steps',
            'role_tier','company_tier','is_referral','is_portal','is_recent_grad',
            'tech_x_exp','tech_x_role','tech_over_apt','exp_sq','age_minus_exp']
cat_cols = ['institute_n','degree','major','current_city','applied_role','company_type',
            'company_size','channel_n','last_job_change','currently_enrolled',
            'overtime_history','awards']
feats = num_cols + cat_cols

# Fill missing on full sets
train_f['is_winner'] = train_f[ID].astype(str).str.strip().isin(win_ids).astype(int)
dev_f['is_winner']   = dev_f[ID].astype(str).str.strip().isin(win_ids).astype(int)
test_f['is_winner']  = 0

def prep_X(df):
    X = df[feats].copy()
    for c in cat_cols: X[c] = X[c].fillna('missing').astype(str)
    for c in num_cols: X[c] = X[c].fillna(train_f[c].median())
    return X

X_tr_full = prep_X(train_f)
X_dv_full = prep_X(dev_f)
X_te_full = prep_X(test_f)

HIGH = np.quantile(train_f[TARGET], 0.95)
print(f"Winner score threshold: {HIGH:.2f}")

# --- 5-fold CV on dev winners ---
np.random.seed(SEED)
win_idx = dev_f.index[dev_f['is_winner'] == 1].values.copy()
np.random.shuffle(win_idx)
folds = np.array_split(win_idx, 5)
non_win_idx = dev_f.index[dev_f['is_winner'] == 0].values

print("\n5-fold CV (train on 20k + 120 winners, eval on 30 held-out + all non-winners):\n")
fold_scores = []
for k, holdout in enumerate(folds):
    train_wins = np.concatenate([folds[j] for j in range(5) if j != k])

    X_aug = pd.concat([X_tr_full, X_dv_full.loc[train_wins]], ignore_index=True)
    y_aug = np.concatenate([train_f[TARGET].values, np.full(len(train_wins), HIGH)])
    w_aug = np.concatenate([np.ones(len(train_f)), np.full(len(train_wins), 5.0)])

    m = CatBoostRegressor(iterations=700, learning_rate=0.05, depth=7,
                          verbose=0, random_seed=SEED, l2_leaf_reg=4)
    m.fit(X_aug, y_aug, sample_weight=w_aug, cat_features=cat_cols)

    eval_idx = np.concatenate([holdout, non_win_idx])
    preds = m.predict(X_dv_full.loc[eval_idx])
    order = np.argsort(-preds)
    top150 = order[:150]
    hit = np.isin(top150, np.arange(len(holdout))).sum()
    fold_scores.append(hit)
    print(f"  Fold {k}: {hit}/30 winners in top 150  → scaled {hit/30*150:.0f}/150")

mean_hit = np.mean(fold_scores)
print(f"\nMEAN CV recall: {mean_hit:.1f}/30 per fold  →  ~{mean_hit/30*150:.0f}/150")