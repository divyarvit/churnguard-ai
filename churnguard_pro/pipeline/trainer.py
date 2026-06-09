import pandas as pd
import numpy as np
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              roc_curve, precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance

MODEL_DIR = os.path.join(BASE, 'models')

TELCO_NUM = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
             'AvgMonthlyCharge','ChargeRatio','ServiceCount',
             'ContractRisk','HighValue','AutoPay']
TELCO_CAT = ['gender','Partner','Dependents','PhoneService','MultipleLines',
             'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
             'TechSupport','StreamingTV','StreamingMovies',
             'Contract','PaperlessBilling','PaymentMethod']
BANK_NUM  = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
             'HasCrCard','IsActiveMember','EstimatedSalary',
             'BalancePerProduct','SalaryBalanceRatio','ZeroBalance',
             'EngagementScore','GoodCredit','HighValue']
BANK_CAT  = ['Geography','Gender']

def load_telco(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df.drop(columns=['customerID'], inplace=True)
    return df

def engineer_telco(df):
    df = df.copy()
    df['AvgMonthlyCharge'] = np.where(df['tenure']>0, df['TotalCharges']/df['tenure'], df['MonthlyCharges'])
    df['ChargeRatio']      = df['MonthlyCharges'] / (df['AvgMonthlyCharge'] + 1)
    svc = ['PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['ServiceCount']  = df[svc].apply(lambda r: sum(1 for v in r if v=='Yes'), axis=1)
    df['ContractRisk']  = df['Contract'].map({'Month-to-month':2,'One year':1,'Two year':0})
    df['HighValue']     = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    df['AutoPay']       = df['PaymentMethod'].str.contains('automatic').astype(int)
    df['TenureBucket']  = pd.cut(df['tenure'], bins=[-1,12,24,48,72],
                                  labels=['<1yr','1-2yr','2-4yr','4+yr']).astype(str)
    return df

def load_bank(path):
    df = pd.read_csv(path)
    df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)
    df.rename(columns={'Exited':'Churn'}, inplace=True)
    return df

def engineer_bank(df):
    df = df.copy()
    df['BalancePerProduct']  = df['Balance'] / (df['NumOfProducts']+1)
    df['SalaryBalanceRatio'] = np.where(df['EstimatedSalary']>0, df['Balance']/df['EstimatedSalary'], 0)
    df['ZeroBalance']        = (df['Balance']==0).astype(int)
    df['EngagementScore']    = df['IsActiveMember']*3 + df['HasCrCard'] + df['NumOfProducts']*2
    df['GoodCredit']         = (df['CreditScore']>=700).astype(int)
    df['HighValue']          = (df['Balance'] > df['Balance'].median()).astype(int)
    df['AgeBucket']          = pd.cut(df['Age'], bins=[0,30,40,50,60,100],
                                       labels=['<30','30-40','40-50','50-60','60+']).astype(str)
    df['TenureBucket']       = pd.cut(df['Tenure'], bins=[-1,2,5,7,10],
                                       labels=['0-2yr','3-5yr','6-7yr','8+yr']).astype(str)
    return df

def oversample(X, y):
    df = pd.DataFrame(X); df['__y'] = y.values
    maj = df[df['__y']==0]; mn = df[df['__y']==1]
    mn_up = resample(mn, replace=True, n_samples=len(maj), random_state=42)
    bal = pd.concat([maj, mn_up]).sample(frac=1, random_state=42)
    return bal.drop('__y',axis=1).values, bal['__y'].values

def build_preprocessor(num_feats, cat_feats):
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)])

def train_pipeline(dataset='telco'):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n{'='*52}\n  Training — {dataset.upper()}\n{'='*52}")
    if dataset == 'telco':
        df = engineer_telco(load_telco(os.path.join(BASE,'data','telco_churn.csv')))
        num_feats, cat_feats, drop_extra = TELCO_NUM, TELCO_CAT, ['TenureBucket']
    else:
        df = engineer_bank(load_bank(os.path.join(BASE,'data','bank_churn.csv')))
        num_feats, cat_feats, drop_extra = BANK_NUM, BANK_CAT, ['AgeBucket','TenureBucket']

    df_full   = df.copy()
    avail_num = [c for c in num_feats if c in df.columns]
    avail_cat = [c for c in cat_feats if c in df.columns]
    X = df.drop(columns=['Churn']+drop_extra, errors='ignore')
    y = df['Churn']
    print(f"  Records: {len(X):,} | Features: {X.shape[1]} | Churn: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessor(avail_num, avail_cat)
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp  = preprocessor.transform(X_test)
    # Replace any remaining NaNs
    X_train_pp = np.nan_to_num(X_train_pp, nan=0.0)
    X_test_pp  = np.nan_to_num(X_test_pp,  nan=0.0)
    X_bal, y_bal = oversample(X_train_pp, y_train)
    print(f"  After oversampling: {len(X_bal):,}")

    try:
        cat_enc = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(avail_cat)
        feature_names = avail_num + list(cat_enc)
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train_pp.shape[1])]

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.1, class_weight='balanced'),
    }
    if HAS_XGB:
        scale_pos = int((y_bal==0).sum()) / max(int((y_bal==1).sum()), 1)
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0
        )
        print("  XGBoost detected and added to model comparison ✅")
    else:
        print("  XGBoost not installed — run: pip install xgboost")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        print(f"\n  [{name}]")
        model.fit(X_bal, y_bal)
        y_pred  = model.predict(X_test_pp)
        y_proba = model.predict_proba(X_test_pp)[:,1]
        cv_res  = cross_validate(model, X_bal, y_bal, cv=cv, scoring=['roc_auc','f1'], return_train_score=True)
        fpr,tpr,_ = roc_curve(y_test, y_proba)
        pc,rc,_   = precision_recall_curve(y_test, y_proba)
        cm        = confusion_matrix(y_test, y_pred)
        results[name] = {
            'accuracy': round(accuracy_score(y_test,y_pred),4),
            'roc_auc':  round(roc_auc_score(y_test,y_proba),4),
            'precision':round(precision_score(y_test,y_pred,zero_division=0),4),
            'recall':   round(recall_score(y_test,y_pred,zero_division=0),4),
            'f1':       round(f1_score(y_test,y_pred,zero_division=0),4),
            'avg_precision':  round(average_precision_score(y_test,y_proba),4),
            'cv_auc_mean':    round(cv_res['test_roc_auc'].mean(),4),
            'cv_auc_std':     round(cv_res['test_roc_auc'].std(),4),
            'cv_f1_mean':     round(cv_res['test_f1'].mean(),4),
            'train_auc_mean': round(cv_res['train_roc_auc'].mean(),4),
            'confusion_matrix': cm.tolist(),
            'fpr':fpr.tolist(),'tpr':tpr.tolist(),
            'precision_curve':pc.tolist(),'recall_curve':rc.tolist(),
        }
        print(f"    Acc={results[name]['accuracy']:.3f} | AUC={results[name]['roc_auc']:.3f} | F1={results[name]['f1']:.3f} | CV={results[name]['cv_auc_mean']:.3f}±{results[name]['cv_auc_std']:.3f}")

    best_name  = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = models[best_name]
    print(f"\n  Best: {best_name} (AUC={results[best_name]['roc_auc']:.4f})")

    if hasattr(best_model,'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=feature_names[:len(best_model.feature_importances_)])
    else:
        perm = permutation_importance(best_model, X_test_pp, y_test, n_repeats=10, random_state=42)
        fi   = pd.Series(perm.importances_mean, index=feature_names[:len(perm.importances_mean)])
    fi = fi.sort_values(ascending=False)

    X_all = df_full.drop(columns=['Churn']+drop_extra, errors='ignore')
    X_all_pp = np.nan_to_num(preprocessor.transform(X_all), nan=0.0)
    probs = best_model.predict_proba(X_all_pp)[:,1]*100
    df_scored = df_full.copy()
    df_scored['ChurnProbability'] = probs.round(1)
    df_scored['PredictedChurn']   = (probs>50).astype(int)
    df_scored['RiskLevel'] = pd.cut(probs,bins=[0,30,60,80,100],labels=['Safe','At-Risk','High Risk','Critical'])

    pkg = dict(best_model=best_model, preprocessor=preprocessor,
               feature_names=feature_names, feature_importance=fi,
               model_results=results, best_model_name=best_name,
               num_feats=avail_num, cat_feats=avail_cat, dataset=dataset,
               train_size=len(X_train), test_size=len(X_test),
               churn_rate=float(y.mean()), n_records=len(X),
               scored_df=df_scored, df_full=df_full)
    path = os.path.join(MODEL_DIR, f'{dataset}_pipeline.pkl')
    with open(path,'wb') as f: pickle.dump(pkg,f)
    print(f"  Saved → {path}")
    return pkg

def load_pipeline(dataset='telco'):
    path = os.path.join(MODEL_DIR,f'{dataset}_pipeline.pkl')
    if not os.path.exists(path): return None
    with open(path,'rb') as f: return pickle.load(f)

def predict_single(input_dict, dataset='telco'):
    pkg = load_pipeline(dataset)
    if not pkg: return None, None
    model,prep = pkg['best_model'],pkg['preprocessor']
    row = {c:0 for c in pkg['num_feats']+pkg['cat_feats']}
    for k,v in input_dict.items():
        if k in row: row[k]=v
    X = pd.DataFrame([row])
    for c in pkg['cat_feats']: X[c]=X[c].astype(str)
    X_pp = np.nan_to_num(prep.transform(X), nan=0.0)
    prob = model.predict_proba(X_pp)[0][1]*100
    return round(float(prob),1), int(prob>50)

if __name__=='__main__':
    train_pipeline('telco')
    train_pipeline('bank')
