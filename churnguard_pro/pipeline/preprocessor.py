import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  TELCO DATASET PIPELINE
# ─────────────────────────────────────────────

def load_telco(path):
    df = pd.read_csv(path)

    # Fix TotalCharges (blank strings for new customers)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Drop ID column
    df.drop(columns=['customerID'], inplace=True)

    return df


def engineer_telco(df):
    df = df.copy()

    # Avg monthly charge over tenure
    df['AvgMonthlyCharge'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    # Charge increase ratio
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['AvgMonthlyCharge'] + 1)

    # Service count — number of add-on services subscribed
    service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['ServiceCount'] = df[service_cols].apply(
        lambda r: sum(1 for v in r if v == 'Yes'), axis=1
    )

    # Contract risk — month-to-month is highest churn risk
    df['ContractRisk'] = df['Contract'].map({
        'Month-to-month': 2,
        'One year': 1,
        'Two year': 0
    })

    # Tenure bucket
    df['TenureBucket'] = pd.cut(df['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['New (<1yr)', 'Growing (1-2yr)', 'Established (2-4yr)', 'Loyal (4+yr)']
    ).astype(str)

    # High-value flag
    df['HighValue'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    # Auto payment (lower churn risk)
    df['AutoPay'] = df['PaymentMethod'].str.contains('automatic').astype(int)

    return df


def get_telco_features():
    numeric = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
               'AvgMonthlyCharge','ChargeRatio','ServiceCount',
               'ContractRisk','HighValue','AutoPay']
    categorical = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                   'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies',
                   'Contract','PaperlessBilling','PaymentMethod']
    return numeric, categorical


# ─────────────────────────────────────────────
#  BANK DATASET PIPELINE
# ─────────────────────────────────────────────

def load_bank(path):
    df = pd.read_csv(path)
    df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)
    df.rename(columns={'Exited': 'Churn'}, inplace=True)
    return df


def engineer_bank(df):
    df = df.copy()

    # Balance per product
    df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)

    # Salary to balance ratio
    df['SalaryBalanceRatio'] = np.where(
        df['EstimatedSalary'] > 0,
        df['Balance'] / df['EstimatedSalary'],
        0
    )

    # Zero balance flag (dormant account)
    df['ZeroBalance'] = (df['Balance'] == 0).astype(int)

    # Age bucket
    df['AgeBucket'] = pd.cut(df['Age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['<30', '30-40', '40-50', '50-60', '60+']
    ).astype(str)

    # Engagement score
    df['EngagementScore'] = (
        df['IsActiveMember'] * 3 +
        df['HasCrCard'] * 1 +
        df['NumOfProducts'] * 2
    )

    # Tenure bucket
    df['TenureBucket'] = pd.cut(df['Tenure'],
        bins=[-1, 2, 5, 7, 10],
        labels=['New (0-2yr)', 'Mid (3-5yr)', 'Established (6-7yr)', 'Loyal (8+yr)']
    ).astype(str)

    # High credit score flag
    df['GoodCredit'] = (df['CreditScore'] >= 700).astype(int)

    # High value customer
    df['HighValue'] = (df['Balance'] > df['Balance'].median()).astype(int)

    return df


def get_bank_features():
    numeric = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
               'HasCrCard','IsActiveMember','EstimatedSalary',
               'BalancePerProduct','SalaryBalanceRatio','ZeroBalance',
               'EngagementScore','GoodCredit','HighValue']
    categorical = ['Geography','Gender']
    return numeric, categorical


# ─────────────────────────────────────────────
#  SHARED PREPROCESSOR
# ─────────────────────────────────────────────

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])
    return preprocessor


def prepare_dataset(dataset='telco'):
    if dataset == 'telco':
        df = load_telco('/home/claude/churnguard_pro/data/telco_churn.csv')
        df = engineer_telco(df)
        numeric, categorical = get_telco_features()
        drop_cols = ['TenureBucket']
    else:
        df = load_bank('/home/claude/churnguard_pro/data/bank_churn.csv')
        df = engineer_bank(df)
        numeric, categorical = get_bank_features()
        drop_cols = ['AgeBucket','TenureBucket']

    # Drop engineered string bucket columns (used for viz only)
    for c in drop_cols:
        if c in df.columns:
            df_viz = df.copy()

    X = df.drop(columns=['Churn'] + drop_cols, errors='ignore')
    y = df['Churn']

    # Re-derive features from X
    num_feats = [c for c in numeric if c in X.columns]
    cat_feats = [c for c in categorical if c in X.columns]

    preprocessor = build_preprocessor(num_feats, cat_feats)
    return X, y, preprocessor, num_feats, cat_feats, df
