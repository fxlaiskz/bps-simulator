import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="需求预测 + BPS协同生产决策", layout="wide")
st.title("📈 需求预测（Prophet）与 BPS 生产决策联合模拟器")
st.markdown("上传主产品(A)和副产品(B)的历史需求数据（CSV或Excel），Prophet 自动预测需求均值与残差标准差，并驱动 BPS 最优产量决策。")

# ----------------------------- BPS 模型函数 -----------------------------
def expected_sales(Q, mu, sigma):
    if sigma <= 0:
        return min(Q, mu)
    z = (Q - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return mu * Phi - sigma * phi + Q * (1 - Phi)

def profit_no_bps(Qa, Qb, params):
    p_a, p_b = params['p_a'], params['p_b']
    c_a = params['c_a']
    d = params['d']
    c_b_total = params['c_b_p'] + params['c_b_m']
    mu_a, sigma_a = params['mu_a'], params['sigma_a']
    mu_b, sigma_b = params['mu_b'], params['sigma_b']
    revenue_a = p_a * expected_sales(Qa, mu_a, sigma_a)
    revenue_b = p_b * expected_sales(Qb, mu_b, sigma_b)
    cost_a = c_a * Qa
    disposal_cost = d * Qa
    cost_b = c_b_total * Qb
    return revenue_a + revenue_b - cost_a - disposal_cost - cost_b

def profit_with_bps(Qa, Qb, params):
    p_a, p_b = params['p_a'], params['p_b']
    c_a = params['c_a']
    c_b_m = params['c_b_m']
    d = params['d']
    c_b_p = params['c_b_p']
    rho = params['rho']
    mu_a, sigma_a = params['mu_a'], params['sigma_a']
    mu_b, sigma_b = params['mu_b'], params['sigma_b']
    revenue_a = p_a * expected_sales(Qa, mu_a, sigma_a)
    revenue_b = p_b * expected_sales(Qb, mu_b, sigma_b)
    cost_a = c_a * Qa
    cost_b_m = c_b_m * Qb
    disposal_cost = d * max(Qa - Qb / rho, 0) if rho > 0 else d * Qa
    extra_procurement = c_b_p * max(Qb - rho * Qa, 0) if rho > 0 else c_b_p * Qb
    return revenue_a + revenue_b - cost_a - cost_b_m - disposal_cost - extra_procurement

def optimize_no_bps(params):
    mu_a, sigma_a = params['mu_a'], params['sigma_a']
    mu_b, sigma_b = params['mu_b'], params['sigma_b']
    p_a, p_b = params['p_a'], params['p_b']
    c_a = params['c_a']
    d = params['d']
    c_b_total = params['c_b_p'] + params['c_b_m']
    if p_a > 0:
        total_cost_a = c_a + d
        fractile_a = (p_a - total_cost_a) / p_a
        fractile_a = np.clip(fractile_a, 0.001, 0.999)
        Qa_opt = stats.norm.ppf(fractile_a, mu_a, sigma_a) if sigma_a > 0 else mu_a * fractile_a
    else:
        Qa_opt = 0
    if p_b > 0:
        fractile_b = (p_b - c_b_total) / p_b
        fractile_b = np.clip(fractile_b, 0.001, 0.999)
        Qb_opt = stats.norm.ppf(fractile_b, mu_b, sigma_b) if sigma_b > 0 else mu_b * fractile_b
    else:
        Qb_opt = 0
    Qa_opt = max(Qa_opt, 0)
    Qb_opt = max(Qb_opt, 0)
    profit = profit_no_bps(Qa_opt, Qb_opt, params)
    return Qa_opt, Qb_opt, profit

def optimize_with_bps(params):
    def objective(x):
        return -profit_with_bps(x[0], x[1], params)
    bounds = [(0, None), (0, None)]
    Qa0, Qb0, _ = optimize_no_bps(params)
    Qa0 = max(Qa0, 1e-2)
    Qb0 = max(Qb0, 1e-2)
    try:
        res = optimize.minimize(objective, [Qa0, Qb0], bounds=bounds, method='L-BFGS-B')
        if res.success:
            Qa_opt, Qb_opt = res.x
        else:
            res = optimize.minimize(objective, [Qa0, Qb0], bounds=bounds, method='Nelder-Mead')
            Qa_opt, Qb_opt = res.x
    except:
        Qa_opt, Qb_opt, _ = optimize_no_bps(params)
    profit = profit_with_bps(Qa_opt, Qb_opt, params)
    return Qa_opt, Qb_opt, profit

# ----------------------------- 辅助函数 -----------------------------
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error(f"不支持的文件类型: {file_extension}")
            return None
        if df.shape[1] < 2:
            st.error("文件必须至少包含两列：日期列和需求量列")
            return None
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

def train_prophet_plotly(df, product_name, periods=1):
    df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    model = Prophet(interval_width=0.8, yearly_seasonality='auto', weekly_seasonality=False, daily_seasonality=False)
    model.add_country_holidays(country_name='CN')
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, include_history=True)
    forecast = model.predict(future)

    mu = forecast['yhat'].iloc[-1]
    train_forecast = model.predict(df)
    residuals = df['y'].values - train_forecast['yhat'].values[:len(df)]
    sigma = np.std(residuals)

    hist_forecast = forecast[forecast['ds'].isin(df['ds'])]
    future_forecast = forecast[~forecast['ds'].isin(df['ds'])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='历史真实值',
                             marker=dict(color='blue', size=8),
                             hovertemplate='日期: %{x}<br>需求量: %{y}<extra></extra>'))
    fig.add_trace(go.Scatter(x=hist_forecast['ds'], y=hist_forecast['yhat'], mode='lines', name='历史拟合值',
                             line=dict(color='blue', width=1.5),
                             hovertemplate='日期: %{x}<br>拟合值: %{y}<extra></extra>'))
    fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='markers', name='未来预测值',
                             marker=dict(color='red', size=10, symbol='circle'),
                             hovertemplate='日期: %{x}<br>预测值: %{y}<extra></extra>'))
    fig.update_layout(title=f'{product_name} 需求预测（未来{periods}期）',
                      xaxis_title='日期', yaxis_title='需求量',
                      hovermode='closest', template='plotly_white')
    fig.update_xaxes(tickangle=45)
    return fig, mu, sigma, forecast

# ----------------------------- 界面布局 -----------------------------
with st.sidebar:
    st.header("⚙️ 参数设置")
    with st.expander("📊 产品参数与历史数据", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### 产品A (主产品)")
            file_a = st.file_uploader("历史数据 (CSV/Excel)", type=["csv", "xlsx", "xls"], key="file_a")
            st.caption("第一列日期，第二列需求量")
            price_a = st.number_input("市场价格 p_a", value=1.00, step=0.05, format="%.2f", key="price_a")
            procurement_a = st.number_input("采购成本 c_a^p", value=0.20, step=0.01, format="%.2f", key="proc_a")
            manufacturing_a = st.number_input("加工成本 c_a^m", value=0.30, step=0.01, format="%.2f", key="manuf_a")
        with colB:
            st.markdown("#### 产品B (副产品)")
            file_b = st.file_uploader("历史数据 (CSV/Excel)", type=["csv", "xlsx", "xls"], key="file_b")
            st.caption("第一列日期，第二列需求量")
            price_b = st.number_input("市场价格 p_b", value=0.80, step=0.05, format="%.2f", key="price_b")
            procurement_b = st.number_input("采购成本 c_b^p", value=0.30, step=0.01, format="%.2f", key="proc_b")
            manufacturing_b = st.number_input("加工成本 c_b^m", value=0.20, step=0.01, format="%.2f", key="manuf_b")
        st.markdown("---")
        st.markdown("#### 公共参数")
        disposal_cost = st.number_input("产品A 废料处置成本 d", value=0.20, step=0.01, format="%.2f", key="disposal")
        rho = st.number_input("BPS 技术转化率 ρ (当前值)", value=0.80, min_value=0.01, max_value=2.00, step=0.01, format="%.2f", key="rho")
        # 新增：任意转化率滑块
        rho_custom = st.slider("🎯 任意转化率 ρ (查看敏感点)", min_value=0.1, max_value=2.0, value=float(rho), step=0.01, key="rho_custom")

if file_a is not None and file_b is not None:
    df_a = load_data(file_a)
    df_b = load_data(file_b)
    if df_a is None or df_b is None:
        st.error("文件读取失败，请检查格式")
        st.stop()

    with st.spinner("训练 Prophet 模型..."):
        fig_a, mu_a, sigma_a, forecast_a = train_prophet_plotly(df_a, "主产品A", periods=1)
        fig_b, mu_b, sigma_b, forecast_b = train_prophet_plotly(df_b, "副产品B", periods=1)

    st.subheader("🔮 Prophet 需求预测")
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        st.plotly_chart(fig_a, use_container_width=True)
    with col_pred2:
        st.plotly_chart(fig_b, use_container_width=True)

    # 显示预测均值与标准差
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("产品A 预测需求均值 μ", f"{mu_a:.2f}")
    col_m1.metric("产品A 需求标准差 σ（历史残差）", f"{sigma_a:.2f}")
    col_m2.metric("产品B 预测需求均值 μ", f"{mu_b:.2f}")
    col_m2.metric("产品B 需求标准差 σ（历史残差）", f"{sigma_b:.2f}")

    # 基础参数（不包含rho_custom，用于表格和后续敏感性分析基础）
    params_base = {
        'p_a': price_a, 'p_b': price_b,
        'c_a': procurement_a + manufacturing_a,
        'c_b_p': procurement_b,
        'c_b_m': manufacturing_b,
        'd': disposal_cost,
        'rho': rho,  # 主rho值，用于表格
        'mu_a': mu_a, 'sigma_a': sigma_a,
        'mu_b': mu_b, 'sigma_b': sigma_b
    }

    # 当前参数下的最优决策（使用主rho）
    with st.spinner("求解当前最优产量..."):
        Qa_none, Qb_none, profit_none = optimize_no_bps(params_base)
        Qa_bps, Qb_bps, profit_bps = optimize_with_bps(params_base)
    improvement = (profit_bps - profit_none) / profit_none * 100 if profit_none > 0 else 0

    # 表格1：产品A/B的关键指标
    table1_data = {
        "指标": ["预测需求均值 μ", "需求标准差 σ", "无BPS产量", "有BPS产量", "无BPS期望利润", "有BPS期望利润"],
        "产品A": [f"{mu_a:.2f}", f"{sigma_a:.2f}", f"{Qa_none:.2f}", f"{Qa_bps:.2f}", f"{profit_none:.2f}", f"{profit_bps:.2f}"],
        "产品B": [f"{mu_b:.2f}", f"{sigma_b:.2f}", f"{Qb_none:.2f}", f"{Qb_bps:.2f}", f"{profit_none:.2f}", f"{profit_bps:.2f}"]
    }
    df_table1 = pd.DataFrame(table1_data)
    st.subheader("📊 产品预测与决策指标")
    st.dataframe(df_table1, use_container_width=True, hide_index=True)

    # 表格2：成本与利润对比
    disposal_none = params_base['d'] * Qa_none
    procurement_none = params_base['c_b_p'] * Qb_none
    disposal_bps = params_base['d'] * max(Qa_bps - Qb_bps / rho, 0) if rho > 0 else params_base['d'] * Qa_bps
    procurement_bps = params_base['c_b_p'] * max(Qb_bps - rho * Qa_bps, 0) if rho > 0 else params_base['c_b_p'] * Qb_bps

    table2_data = {
        "场景": ["无BPS", "有BPS"],
        "处置成本": [f"{disposal_none:.2f}", f"{disposal_bps:.2f}"],
        "采购成本": [f"{procurement_none:.2f}", f"{procurement_bps:.2f}"],
        "期望利润": [f"{profit_none:.2f}", f"{profit_bps:.2f}"]
    }
    df_table2 = pd.DataFrame(table2_data)
    st.subheader("💰 成本与利润对比")
    st.dataframe(df_table2, use_container_width=True, hide_index=True)
    st.metric("利润改进百分比", f"{improvement:.2f}%", delta="BPS技术收益")

    # ----------------------------- 敏感性分析（增加自定义ρ点） -----------------------------
    st.subheader("📉 BPS 技术转化率 ρ 的敏感性分析")
    # 使用较密的点绘制曲线
    rho_range = np.linspace(0.1, 1.5, 200)
    Qa_none_list, Qb_none_list, Qa_bps_list, Qb_bps_list = [], [], [], []
    profit_none_list, profit_bps_list = [], []

    for rho_val in rho_range:
        params_temp = params_base.copy()
        params_temp['rho'] = rho_val
        qa_n, qb_n, p_n = optimize_no_bps(params_temp)
        qa_b, qb_b, p_b = optimize_with_bps(params_temp)
        Qa_none_list.append(qa_n); Qb_none_list.append(qb_n)
        Qa_bps_list.append(qa_b); Qb_bps_list.append(qb_b)
        profit_none_list.append(p_n); profit_bps_list.append(p_b)

    # 单独计算自定义 ρ 下的最优决策
    params_custom = params_base.copy()
    params_custom['rho'] = rho_custom
    Qa_none_custom, Qb_none_custom, profit_none_custom = optimize_no_bps(params_custom)
    Qa_bps_custom, Qb_bps_custom, profit_bps_custom = optimize_with_bps(params_custom)

    # 产量对比图（左右子图）
    fig_prod = make_subplots(rows=1, cols=2, subplot_titles=("产品A产量对比", "产品B产量对比"))
    # 产品A曲线
    fig_prod.add_trace(go.Scatter(x=rho_range, y=Qa_none_list, mode='lines', name='无BPS 产品A',
                                  line=dict(color='blue', dash='dash')), row=1, col=1)
    fig_prod.add_trace(go.Scatter(x=rho_range, y=Qa_bps_list, mode='lines', name='有BPS 产品A',
                                  line=dict(color='blue', dash='solid')), row=1, col=1)
    # 产品B曲线
    fig_prod.add_trace(go.Scatter(x=rho_range, y=Qb_none_list, mode='lines', name='无BPS 产品B',
                                  line=dict(color='red', dash='dash')), row=1, col=2)
    fig_prod.add_trace(go.Scatter(x=rho_range, y=Qb_bps_list, mode='lines', name='有BPS 产品B',
                                  line=dict(color='red', dash='solid')), row=1, col=2)
    # 标记自定义 ρ 点（产品A和产品B的有BPS产量）
    fig_prod.add_trace(go.Scatter(x=[rho_custom], y=[Qa_bps_custom], mode='markers',
                                  marker=dict(color='black', size=10, symbol='star'),
                                  name=f'ρ={rho_custom:.2f} 产品A有BPS'), row=1, col=1)
    fig_prod.add_trace(go.Scatter(x=[rho_custom], y=[Qb_bps_custom], mode='markers',
                                  marker=dict(color='black', size=10, symbol='star'),
                                  name=f'ρ={rho_custom:.2f} 产品B有BPS'), row=1, col=2)
    fig_prod.update_xaxes(title_text="BPS转化率 ρ", row=1, col=1)
    fig_prod.update_xaxes(title_text="BPS转化率 ρ", row=1, col=2)
    fig_prod.update_yaxes(title_text="最优产量", row=1, col=1)
    fig_prod.update_yaxes(title_text="最优产量", row=1, col=2)
    fig_prod.update_layout(height=500, width=900, title_text="产量随ρ变化对比", showlegend=True)

    # 利润对比图
    fig_profit = go.Figure()
    fig_profit.add_trace(go.Scatter(x=rho_range, y=profit_none_list, mode='lines', name='无BPS期望利润',
                                    line=dict(color='green', dash='dash')))
    fig_profit.add_trace(go.Scatter(x=rho_range, y=profit_bps_list, mode='lines', name='有BPS期望利润',
                                    line=dict(color='green', dash='solid')))
    fig_profit.add_trace(go.Scatter(x=[rho_custom], y=[profit_bps_custom], mode='markers',
                                    marker=dict(color='black', size=12, symbol='star'),
                                    name=f'ρ={rho_custom:.2f} 有BPS利润'))
    fig_profit.update_layout(title="有无BPS技术的期望利润对比", xaxis_title="BPS转化率 ρ", yaxis_title="期望利润",
                             height=500, width=900, showlegend=True)

    # 显示图表
    st.plotly_chart(fig_prod, use_container_width=True)
    st.plotly_chart(fig_profit, use_container_width=True)

    # 显示自定义 ρ 下的具体数值
    st.subheader(f"🎯 转化率 ρ = {rho_custom:.2f} 时的详细结果")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("无BPS - 产品A产量", f"{Qa_none_custom:.2f}")
        st.metric("无BPS - 产品B产量", f"{Qb_none_custom:.2f}")
        st.metric("无BPS - 期望利润", f"{profit_none_custom:.2f}")
    with col_c2:
        st.metric("有BPS - 产品A产量", f"{Qa_bps_custom:.2f}", delta=f"{Qa_bps_custom - Qa_none_custom:.2f}")
        st.metric("有BPS - 产品B产量", f"{Qb_bps_custom:.2f}", delta=f"{Qb_bps_custom - Qb_none_custom:.2f}")
        st.metric("有BPS - 期望利润", f"{profit_bps_custom:.2f}", delta=f"{profit_bps_custom - profit_none_custom:.2f}")
    with col_c3:
        improvement_custom = (profit_bps_custom - profit_none_custom) / profit_none_custom * 100 if profit_none_custom > 0 else 0
        st.metric("利润改进百分比", f"{improvement_custom:.2f}%")
        if rho_custom > 0:
            saved_disposal = params_custom['d'] * max(Qa_bps_custom - Qb_bps_custom / rho_custom, 0)
            saved_procurement = params_custom['c_b_p'] * max(Qb_bps_custom - rho_custom * Qa_bps_custom, 0)
            st.metric("节省处置成本", f"{saved_disposal:.2f}")
            st.metric("节省采购成本", f"{saved_procurement:.2f}")

    # 决策区域判断（基于当前主rho的产量关系）
    if rho > 0:
        eps = 1e-6
        if abs(Qb_bps - rho * Qa_bps) < eps:
            region = "区域2 (废料恰好全部利用)"
        elif Qb_bps < rho * Qa_bps:
            region = "区域1 (废料有剩余，B全部来自废料)"
        else:
            region = "区域3 (废料全部利用且需额外采购)"
        st.info(f"📌 当前决策区域（主ρ={rho:.2f}）：{region}")
else:
    st.info("👈 请在左侧边栏中上传主产品A和副产品B的历史需求文件（CSV或Excel）")

st.markdown("---")
st.caption("模型假设：需求服从正态分布，Prophet 预测值作为均值，历史残差标准差作为波动参数。BPS 模型来自《节能减排政策下面向随机需求的主副产品协同生产决策研究》。")