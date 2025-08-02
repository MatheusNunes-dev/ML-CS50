# %%
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import seaborn as sns
from feature_engine.encoding import OneHotEncoder 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import pipeline
from sklearn import metrics 

class ValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.login_mapping = {"Phone": "Mobile Phone"}
        self.payment_mapping = {"COD": "Cash on Delivery", "CC": "Credit Card"}
        self.order_mapping = {"Mobile": "Mobile Phone"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['PreferredLoginDevice'] = X['PreferredLoginDevice'].replace(self.login_mapping)
        X['PreferredPaymentMode'] = X['PreferredPaymentMode'].replace(self.payment_mapping)
        X['PreferedOrderCat'] = X['PreferedOrderCat'].replace(self.order_mapping)
        return X


# %%

df = pd.read_csv("ecommerce.csv", sep=";")
df.head()
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

# %%
oot = df[df['CustomerID'] >= 55200].copy()
oot.shape  

# %%
df_train = df[df['CustomerID'] < 55200].copy()
df_train.shape

# %%
features = df_train.columns[2:]
target = 'Churn'

X, y = df_train[features], df_train[target]

# %%
X_oot = oot[features]
y_oot = oot[target]


# %%
# SAMPLE
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)


# %%
print("Taxa variável resposta Treino:", y.mean())
print("Taxa variável resposta Treino:", y_train.mean())
print("Taxa variável resposta Teste:", y_test.mean())

# %%
# EXPLORE

X_train.info()

# %%
round((X_train.isnull().sum() / X_train.shape[0]) * 100, 2)


# %%

# Variaveis Categoricas
sections_features = [X_train['PreferredLoginDevice'].unique(),
                     X_train['PreferredPaymentMode'].unique(),
                     X_train['Gender'].unique(),
                     X_train['PreferedOrderCat'].unique(),
                     X_train['MaritalStatus'].unique()
                     ]

for i in sections_features:
    print(i)

# %%
X_train = ValueReplacer().fit_transform(X_train)
X_test = ValueReplacer().fit_transform(X_test)
X_oot = ValueReplacer().fit_transform(X_oot)
# %%
col_objects = X_train.select_dtypes(include=object).columns
col_objects

# %%

onehot = OneHotEncoder(variables=['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
       'PreferedOrderCat', 'MaritalStatus'])

onehot.fit(X_train)

X_train_dum = onehot.transform(X_train)
X_test_dum = onehot.transform(X_test)
X_oot_dum = onehot.transform(X_oot)


# %%


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_dum)
X_test_scaled = scaler.transform(X_test_dum)
X_oot_scaled = scaler.transform(X_oot_dum)

# %%
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)
X_oot_imputed = imputer.transform(X_oot_scaled)

# %%
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train_dum.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test_dum.columns, index=X_test.index)
X_oot_imputed = pd.DataFrame(X_oot_imputed, columns=X_oot_dum.columns, index=X_oot.index)

# %%

X_train_real = scaler.inverse_transform(X_train_imputed)
X_train_real = pd.DataFrame(X_train_real, columns=X_train_dum.columns, index=X_train.index)

X_test_real = scaler.inverse_transform(X_test_imputed)
X_test_real = pd.DataFrame(X_test_real, columns=X_test_dum.columns, index=X_test.index)

X_oot_real = scaler.inverse_transform(X_oot_imputed)
X_oot_real = pd.DataFrame(X_oot_real, columns=X_oot_dum.columns, index=X_oot.index)
# %%%

df_analise = X_train_real.copy()
df_analise[target] = y_train
summario = df_analise.groupby(by=target).agg(["mean", "median"]).T
summario['diff_abs'] = summario[0] - summario[1]
summario['diff_rel'] = summario[0] / summario[1]
summario.sort_values(by=['diff_rel'], ascending=False)

# %%

clf = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train_real, y_train)


plt.figure(dpi=400, figsize=[15,6])
tree.plot_tree(clf, feature_names=X_train_real.columns,
               class_names=clf.classes_.astype(str), filled=True)
# %%

features_importances = (pd.Series(clf.feature_importances_, index=X_train_real.columns).sort_values(ascending=False).reset_index())

features_importances['acum'] = features_importances[0].cumsum()
features_importances[features_importances['acum'] < 0.96]

# %%
df_plot = pd.DataFrame({ 
    'Tenure': X_train_real['Tenure'],
    'Churn' : y_train
})

df_plot['TenureGroup'] = pd.cut(df_plot['Tenure'], bins=[0,3,6,9,50], labels=['≤3 months', '4–6 months', '7–9 months', '<= 12 months'], include_lowest=True)
churn_tenure_rate = df_plot.groupby('TenureGroup')['Churn'].mean()

plt.figure(figsize=(15,6))
churn_tenure_rate.plot(kind='bar')
plt.ylim(0, 0.50)
plt.title("Tenure x Churn")
plt.show()

# %%

churn_tenure_plot = sns.countplot(data = df_plot, x = 'TenureGroup', hue = 'Churn', palette = 'Set2')


plt.title('Customer Churn by Tenure Range')
plt.xlabel('Tenure Bins')
plt.ylabel('Frequency')
plt.show()


# %%


df_complain_plot = pd.DataFrame({
    'Complain' :  X_train_real['Complain'],
    'Churn' : y_train
})


churn_complain_rate = df_complain_plot.groupby("Complain")['Churn'].mean()
plt.figure(figsize=(15,6))
churn_complain_rate.plot(kind='bar')
plt.ylim(0, 0.40)
plt.title("Complain x Churn")
plt.show()


# %%

df_day_plot = pd.DataFrame({
    'DaySinceLastOrder' : X_train_real['DaySinceLastOrder'],
    'Churn' : y_train
})


df_day_plot['DaySinceLastOrderGroup'] = pd.cut(df_day_plot['DaySinceLastOrder'], bins=[0,2,4,6,46], labels=['0-2 days', '2-4 days', '4-6 days', '6+ days'], include_lowest=True)
churn_DaySinceLastOrder_rate = df_day_plot.groupby('DaySinceLastOrderGroup')['Churn'].mean()

plt.figure(figsize=(15,6))
churn_DaySinceLastOrder_rate.plot(kind='bar')
plt.ylim(0, 0.40)
plt.title("DaySinceLastOrderGroup x Churn")
plt.show()

# %%
print(df_day_plot['Churn'].mean())

# %%
sns.boxplot(x='Churn', y='DaySinceLastOrder', data=df_day_plot)
plt.title("Distribuição de Days Since Last Order por Churn")
plt.show()
# %%

best_features = features_importances[features_importances['acum'] < 0.96]['index'].tolist()
best_features

# %%
from sklearn import linear_model

reg_imputed = linear_model.LogisticRegression(penalty= None, random_state = 42, max_iter=10000)
reg_imputed.fit(X_train_imputed, y_train)

# %%

# Treino
y_pred_train = reg_imputed.predict(X_train_imputed)
y_proba_train = reg_imputed.predict_proba(X_train_imputed)[:, 1]

train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
train_auc = metrics.roc_auc_score(y_train, y_proba_train)
train_recall = metrics.recall_score(y_train, y_pred_train)
train_precision = metrics.precision_score(y_train, y_pred_train)

print("Train accuracy:", train_accuracy)
print("Train AUC:", train_auc)
print("Train Recall:", train_recall)
print("Train Precision:", train_precision)

# %%

# Teste
y_pred_test = reg_imputed.predict(X_test_imputed)
y_proba_test = reg_imputed.predict_proba(X_test_imputed)[:, 1]

test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
test_auc = metrics.roc_auc_score(y_test, y_proba_test)
test_recall = metrics.recall_score(y_test, y_pred_test)
test_precision = metrics.precision_score(y_test, y_pred_test)

print("Test accuracy:", test_accuracy)
print("Test AUC:", test_auc)
print("Test Recall:", test_recall)
print("Test Precision:", test_precision)

# %%

# Out-of-Time
y_pred_oot = reg_imputed.predict(X_oot_imputed)
y_proba_oot = reg_imputed.predict_proba(X_oot_imputed)[:, 1]

oot_accuracy = metrics.accuracy_score(y_oot, y_pred_oot)
oot_auc = metrics.roc_auc_score(y_oot, y_proba_oot)
oot_recall = metrics.recall_score(y_oot, y_pred_oot)
oot_precision = metrics.precision_score(y_oot, y_pred_oot)

print("OOT accuracy:", oot_accuracy)
print("OOT AUC:", oot_auc)
print("OOT Recall:", oot_recall)
print("OOT Precision:", oot_precision)
# %%

model_pipeline = pipeline.Pipeline( steps=[
    ("Onehot", onehot),
    ("Scaler", scaler),
    ("Imputer", imputer),
    ("Model", reg_imputed)
])

model_pipeline.fit(X_train, y_train)
# %%

model = pd.Series( {"Model": model_pipeline,
                    "features": features,
})

model.to_pickle("ecommerce_pipeline.pkl")
# %%
