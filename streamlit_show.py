import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# 在庫最適化の分析関数
def analyze_inventory_optimization(audience, unit_price, purchase_rate, cost_per_unit, stock_range_max=100000, step=1000):
    stock_range = np.arange(0, stock_range_max, step)
    expected_demand = audience * purchase_rate
    total_demand = expected_demand
    results = []

    for stock in stock_range:
        actual_sales = min(stock, total_demand)
        revenue = actual_sales * unit_price
        procurement_cost = stock * cost_per_unit
        opportunity_loss = max(0, total_demand - stock) * (unit_price - cost_per_unit)
        excess_stock = max(0, stock - total_demand)
        excess_cost = excess_stock * cost_per_unit
        profit = revenue - procurement_cost - excess_cost

        results.append({
            '在庫数': stock,
            '売上': revenue,
            '仕入れコスト': procurement_cost,
            '機会損失': opportunity_loss,
            '余剰在庫コスト': excess_cost,
            '純利益': profit
        })

    df_results = pd.DataFrame(results)
    optimal_stock = df_results.loc[df_results['純利益'].idxmax(), '在庫数']
    max_profit = df_results['純利益'].max()

    return optimal_stock, max_profit, df_results

# Streamlit アプリ
def main():
    st.title("在庫最適化分析ツール")

    df = pd.read_csv(r"results\predictions\decision_tree_predictions.csv")

    # 入力パラメータ
    match_number = st.number_input("何試合目", min_value=1, value=1, step=1) - 1   
    capacity = st.number_input("定員数", min_value=1, value=40142, step=1)
    audience = df.loc[match_number, 'Predicted']*capacity
    unit_price = st.number_input("販売単価 (円)", min_value=1, value=850, step=1)
    purchase_rate = st.number_input("1人当たり購入数", min_value=0.1, value=2.0, step=0.1)
    cost_per_unit = st.number_input("仕入れ単価 (円)", min_value=1, value=200, step=1)
    stock_range_max = st.number_input("最大在庫数", min_value=1000, value=100000, step=1000)
    step = st.number_input("在庫数の刻み幅", min_value=100, value=1000, step=100)

    if st.button("分析を実行"):
        optimal_stock, max_profit, results = analyze_inventory_optimization(
            audience, unit_price, purchase_rate, cost_per_unit, stock_range_max, step
        )

        st.write(f"最適在庫数: {optimal_stock:,.0f}")
        st.write(f"予測最大利益: ¥{max_profit:,.0f}")

        # グラフ描画
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # 純利益
        ax[0].plot(results['在庫数'], results['純利益'], label='純利益', color='green', linewidth=2)
        ax[0].axvline(x=optimal_stock, color='red', linestyle='--', label=f'最適在庫数: {optimal_stock:,.0f}')
        ax[0].axhline(y=max_profit, color='blue', linestyle='--', label=f'最大利益: ¥{max_profit:,.0f}')
        ax[0].set_xlabel('在庫数')
        ax[0].set_ylabel('金額 (円)')
        ax[0].set_title('在庫数と純利益の関係')
        ax[0].grid(True)
        ax[0].legend()

        # コスト要素
        ax[1].plot(results['在庫数'], results['売上'], label='売上', color='blue')
        ax[1].plot(results['在庫数'], results['仕入れコスト'], label='仕入れコスト', color='red')
        ax[1].plot(results['在庫数'], results['機会損失'], label='機会損失', color='orange')
        ax[1].plot(results['在庫数'], results['余剰在庫コスト'], label='余剰在庫コスト', color='purple')
        ax[1].set_xlabel('在庫数')
        ax[1].set_ylabel('金額 (円)')
        ax[1].set_title('在庫数とコスト要素の関係')
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # 分析結果データの表示
        st.write("分析結果データ")
        st.dataframe(results)

if __name__ == "__main__":
    main()