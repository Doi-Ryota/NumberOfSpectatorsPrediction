import pandas as pd
import matplotlib.pyplot as plt
import joblib
import japanize_matplotlib
import shap

results = pd.read_csv("../results/predictions/decision_tree_predictions.csv")
X_test = pd.read_csv("../results/predictions/decision_tree_X_test.csv")

# 実際の観客数 vs 予測観客数
plt.figure(figsize=(8, 6))
plt.scatter(results.loc[:,"Actual"], results.loc[:,"Predicted"], alpha=0.6)
plt.plot([0.8, 1], [0.8, 1], 'r--', lw=2)  # 完璧な予測を示す直線
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Audience')
plt.savefig("../results/plots/Actual_VS_Predicted.png")


# 残差プロット (Residual Plot)
residuals = results.loc[:,"Actual"] - results.loc[:,"Predicted"]
plt.figure(figsize=(8, 6))
plt.scatter(results.loc[:,"Predicted"], residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig("../results/plots/Residual_Plot.png")


# 特徴量重要度の可視化
# モデルをロード
model = joblib.load('../results/models/decision_tree_model.pkl')
feature_importance = model.feature_importances_
# 特徴量の名前と重要度を対応させてDataFrameにする
feature_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=True)
# 可視化
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.savefig("../results/plots/Feature_Importance.png")


# 予測分布の可視化
plt.figure(figsize=(8, 6))
plt.hist(results.loc[:,"Predicted"], bins=30, alpha=0.7, color='g', edgecolor='black')
plt.xlabel('Predicted Audience')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Audience')
plt.savefig("../results/plots/Distribution.png")


# 残差の大きいデータのForcePlotを表示
# SHAP Explainerの作成
explainer = shap.TreeExplainer(model)
# SHAP値の計算
shap_values = explainer.shap_values(X_test)
# データの結合
combined = pd.concat([X_test, results], axis=1)
# 残差の計算
combined["Residuals"] = combined["Actual"] - combined["Predicted"]
# 残差が大きい順に並べ替え
sorted_combined = combined.sort_values(by="Residuals", ascending=False)
# 指定したインスタンスのインデックス
indices = sorted_combined.head(5).index
# Force Plotを表示
for idx in indices:
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx], matplotlib=True)
    plt.savefig(f"../results/plots/Froce_Plot{idx}.png")