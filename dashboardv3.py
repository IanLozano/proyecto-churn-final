#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, dcc, html, Input, Output, State


# Config / Paths
DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
API_URL_BATCH = "http://127.0.0.1:8000/predict_batch"  # endpoint real por lote


# Carga y limpieza (Datos REALES Telco)
df = pd.read_csv(DATA_PATH)

# Asegurar tipos numéricos
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges']   = pd.to_numeric(df['TotalCharges'],   errors='coerce')
df['tenure']         = pd.to_numeric(df['tenure'],         errors='coerce')

# Corrección TotalCharges = 0 si tenure=0 y rellenar NaN con MonthlyCharges*tenure
df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
mask_tc_nan = df['TotalCharges'].isna()
df.loc[mask_tc_nan, 'TotalCharges'] = df.loc[mask_tc_nan, 'MonthlyCharges'] * df.loc[mask_tc_nan, 'tenure']

# Limpieza de categóricas
cat_cols = ['gender','InternetService','StreamingTV','StreamingMovies',
            'Contract','PaymentMethod','PaperlessBilling','Churn']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Variables de trabajo
df['Tenure'] = df['tenure']
df = df[['Churn','MonthlyCharges','Tenure','Contract']].copy()


# Scoring REAL vía API
def call_api_batch(records):
    """
    Llama a la API de modelo para obtener probabilidades de churn.
    records: lista de dicts con Contract, Tenure, MonthlyCharges.
    Devuelve lista de probabilidades o lista de NaN si falla.
    """
    try:
        r = requests.post(API_URL_BATCH, json={"records": records}, timeout=8)
        r.raise_for_status()
        probs = r.json().get("probas", [])
        if not isinstance(probs, list) or len(probs) != len(records):
            # Tamaño inesperado -> devolvemos NaN
            return [np.nan] * len(records)
        return [float(p) if p is not None else np.nan for p in probs]
    except Exception as e:
        # Para debug en consola
        print("Error llamando a la API de batch:", e)
        return [np.nan] * len(records)


def add_risk_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la columna Risk usando EXCLUSIVAMENTE la API.
    Si la API falla, Risk queda en NaN (no se usa nada sintético).
    """
    d = data.copy()
    payload = d[["Contract", "Tenure", "MonthlyCharges"]].to_dict(orient="records")
    probs = call_api_batch(payload)
    d["Risk"] = np.clip(probs, 0, 1.0)
    return d


# Calcula RISK UNA SOLA VEZ (mejor performance)
df = add_risk_column(df)

# KPIs base
def calc_kpis(data: pd.DataFrame):
    churn_rate = data["Churn"].eq("Yes").mean() * 100 if len(data) else 0.0
    avg_monthly = data["MonthlyCharges"].mean() if len(data) else 0.0
    avg_tenure = data["Tenure"].mean() if len(data) else 0.0
    total_customers = len(data)
    return churn_rate, avg_monthly, avg_tenure, total_customers

# Helpers UI
def kpi_card(title, value, color):
    return html.Div([
        html.H4(title, style={"color": "#666", "marginBottom": "0"}),
        html.H2(value, style={"color": color, "marginTop": "5px"})
    ], style={
        "background": "white",
        "borderRadius": "15px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
        "padding": "18px",
        "textAlign": "center",
        "flex": "1",
        "minWidth": "200px"
    })

def fig_pie(data: pd.DataFrame):
    fig = px.pie(data, names="Churn", title="Distribución de Churn",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig

def fig_contract(data: pd.DataFrame):
    fig = px.histogram(data, x="Contract", color="Churn", barmode="group",
                       title="Churn por tipo de contrato",
                       color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    return fig

def fig_monthly(data: pd.DataFrame):
    fig = px.histogram(data, x="MonthlyCharges", color="Churn", nbins=40, opacity=0.75,
                       title="Distribución de MonthlyCharges",
                       color_discrete_sequence=["#EF553B", "#00CC96"])
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    return fig

def fig_box(data: pd.DataFrame):
    fig = px.box(data, x="Churn", y="Tenure", color="Churn",
                 title="Tenure por estado de churn",
                 color_discrete_sequence=["#00CC96", "#EF553B"])
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    return fig

def fig_gauge(prob):
    p = float(np.clip(prob, 0.0, 1.0)) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p,
        number={'suffix': '%', 'valueformat': '.1f'},
        title={'text': "Probabilidad de Churn"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#EF553B"},
            'steps': [
                {'range': [0, 30], 'color': '#E5E7EB'},
                {'range': [30, 60], 'color': '#CBD5E1'},
                {'range': [60, 100], 'color': '#94A3B8'}
            ]
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig

def fig_risk_by_contract(data: pd.DataFrame):
    g = (data.groupby("Contract", as_index=False)
            .agg(Risk=("Risk","mean"), N=("Risk","size"))
            .sort_values("Risk", ascending=False))
    fig = px.bar(
        g, x="Contract", y="Risk",
        text=(g["Risk"]*100).round(1).astype(str)+"%",
        title="Riesgo promedio por tipo de contrato",
        color="Risk", color_continuous_scale="Reds", range_y=[0,1]
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=20,r=20,t=60,b=40), coloraxis_showscale=False)
    return fig

# ===== Heatmap: Mapa de riesgo real =====
def fig_riskmap(data: pd.DataFrame):
    d = data.copy()
    # Bins
    d["Tenure_bin"]  = pd.cut(d["Tenure"], bins=[0,6,12,24,36,48,60,72], include_lowest=True)
    d["Charges_bin"] = pd.cut(d["MonthlyCharges"], bins=[20,40,60,80,100,120], include_lowest=True)

    # Matriz de riesgo promedio
    piv = d.pivot_table(
        index="Tenure_bin",
        columns="Charges_bin",
        values="Risk",
        aggfunc="mean",
        observed=False
    )

    # Etiquetas legibles
    x_labels = [f"{c.left:.0f}-{c.right:.0f}" for c in piv.columns]
    y_labels = [f"{r.left:.0f}-{r.right:.0f}" for r in piv.index]

    z = piv.to_numpy()
    # Enmascarar celdas sin datos
    z = np.where(np.isfinite(z), z, None)

    fig = go.Figure(data=go.Heatmap(
        x=x_labels, y=y_labels, z=z,
        colorscale="Reds", zmin=0, zmax=1,
        colorbar=dict(title="Prob.")
    ))
    fig.update_layout(
        title="Mapa de riesgo (Tenure vs MonthlyCharges)",
        xaxis_title="MonthlyCharges ($)",
        yaxis_title="Tenure (meses)",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig


# App
app = Dash(__name__)
app.title = "Telco Churn Dashboard (Modelo Real)"

tenure_min, tenure_max = int(df["Tenure"].min()), int(df["Tenure"].max())
charge_min, charge_max = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())

app.layout = html.Div([
    html.H1("Telco Customer Churn — Dashboard conectado al modelo",
            style={"textAlign": "center", "color": "#222", "fontFamily": "Segoe UI", "marginBottom": "8px"}),
    html.P("Arriba ves el comportamiento del RIESGO estimado por el modelo (no solo el EDA). Abajo puedes filtrar.",
           style={"textAlign": "center", "color": "#888", "marginTop": "0"}),

    # KPIs
    html.Div(id="kpi-row", style={"display": "flex", "gap": "20px", "margin": "20px", "flexWrap": "wrap"}),

    # Dos columnas
    html.Div([
        # Izquierda: filtros + scoring individual
        html.Div([
            html.Div([
                html.H3("Controles", style={"marginTop": "0", "marginBottom": "10px", "color": "#222"}),

                html.Label("Contract", style={"fontWeight": 600}),
                dcc.Checklist(
                    id="flt-contract",
                    options=[{"label": c, "value": c} for c in df["Contract"].unique()],
                    value=list(df["Contract"].unique()),
                    inputStyle={"marginRight": "6px", "marginLeft": "4px"},
                    labelStyle={"display": "block", "marginBottom": "6px"}
                ),

                html.Label("Estado de Churn", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.Checklist(
                    id="flt-churn",
                    options=[{"label": s, "value": s} for s in ["Yes", "No"]],
                    value=["Yes", "No"],
                    inputStyle={"marginRight": "6px", "marginLeft": "4px"},
                    labelStyle={"display": "inline-block", "marginRight": "12px"}
                ),

                html.Label("Tenure (meses)", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.RangeSlider(
                    id="flt-tenure",
                    min=tenure_min, max=tenure_max, step=1, value=[tenure_min, tenure_max],
                    marks={0:"0", 12:"12", 24:"24", 36:"36", 48:"48", 60:"60", 72:"72"},
                    updatemode="mouseup"
                ),

                html.Label("MonthlyCharges ($)", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.RangeSlider(
                    id="flt-charges",
                    min=round(charge_min, 0), max=round(charge_max, 0), step=1,
                    value=[round(charge_min, 0), round(charge_max, 0)],
                    marks={20:"20", 40:"40", 60:"60", 80:"80", 100:"100", 120:"120"},
                    updatemode="mouseup"
                ),

                html.Hr(),
                html.Label("Gráfico principal", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="chart-type",
                    options=[
                        {"label": "Mapa de riesgo (2D)", "value": "riskmap"},
                        {"label": "Riesgo por contrato (Barras)", "value": "riskcontract"},
                        {"label": "Distribución de Churn (Pie)", "value": "pie"},
                        {"label": "Churn por Contrato (Barras)", "value": "contract"},
                        {"label": "MonthlyCharges (Hist)", "value": "monthly"},
                        {"label": "Tenure por Churn (Box)", "value": "box"},
                    ],
                    value="riskmap",
                    clearable=False
                ),

                html.Hr(),

                html.H3("Scoring (cliente individual)", style={"marginBottom": "8px", "color": "#222"}),
                html.Label("Contract", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="sc-contract",
                    options=[{"label": c, "value": c} for c in ["Month-to-month", "One year", "Two year"]],
                    value="Month-to-month",
                    clearable=False
                ),

                html.Label("Tenure (meses)", style={"fontWeight": 600, "marginTop": "10px"}),
                dcc.Input(
                    id="sc-tenure",
                    type="number",
                    min=0, max=72, step=1, value=6,
                    style={"width": "100%", "padding": "8px", "borderRadius": "8px", "border": "1px solid #e5e7eb"}
                ),

                html.Label("MonthlyCharges ($)", style={"fontWeight": 600, "marginTop": "10px"}),
                dcc.Input(
                    id="sc-charges",
                    type="number",
                    min=0, step=1, value=70,
                    style={"width": "100%", "padding": "8px", "borderRadius": "8px", "border": "1px solid #e5e7eb"}
                ),

                html.Button("Predecir Churn", id="btn-score",
                            style={"marginTop":"12px","padding":"10px 14px","borderRadius":"12px",
                                   "border":"1px solid #111827","background":"#111827","color":"white",
                                   "cursor":"pointer","fontWeight":600, "width":"100%"}),

                html.Div(style={"display":"flex","gap":"8px","marginTop":"10px"}, children=[
                    html.Button("Reset filtros", id="btn-reset",
                                style={"flex":"1","padding":"8px 12px","borderRadius":"10px",
                                       "border":"1px solid #e5e7eb","background":"white","cursor":"pointer"}),
                    html.Button("Descargar CSV", id="btn-download",
                                style={"flex":"1","padding":"8px 12px","borderRadius":"10px",
                                       "border":"1px solid #111827","background":"#111827","color":"white","cursor":"pointer"})
                ]),
                dcc.Download(id="download-data"),

                html.Div(id="score-msg", style={"color":"#6b7280","fontSize":"12px","marginTop":"6px"})
            ]),
        ], style={
            "background": "white", "borderRadius": "15px", "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            "padding": "18px", "flex": "1", "minWidth": "320px", "maxWidth": "420px"
        }),

        # Derecha: gráfico principal + gauge
        html.Div([
            dcc.Loading(
                dcc.Graph(id="main-graph", style={"height": "420px"}),
                type="circle"
            ),
            html.Div(id="sample-size", style={"textAlign":"right", "color":"#666", "fontSize":"12px"}),
            html.Hr(),
            dcc.Loading(
                dcc.Graph(id="score-gauge", style={"height": "260px"}),
                type="circle"
            )
        ], style={
            "background": "white", "borderRadius": "15px", "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            "padding": "12px", "flex": "2", "minWidth": "360px"
        })

    ], style={"display": "flex", "gap": "20px", "margin": "0 20px 20px 20px", "flexWrap": "wrap"}),

], style={"backgroundColor": "#f8f9fc", "padding": "16px"})

# Filtros y Callbacks
def apply_filters(data: pd.DataFrame, contract_sel, churn_sel, tenure_rng, charge_rng):
    d = data.copy()
    if contract_sel:
        d = d[d["Contract"].isin(contract_sel)]
    if churn_sel:
        d = d[d["Churn"].isin(churn_sel)]
    if tenure_rng and len(tenure_rng) == 2:
        d = d[(d["Tenure"] >= tenure_rng[0]) & (d["Tenure"] <= tenure_rng[1])]
    if charge_rng and len(charge_rng) == 2:
        d = d[(d["MonthlyCharges"] >= charge_rng[0]) & (d["MonthlyCharges"] <= charge_rng[1])]
    return d

@app.callback(
    Output("kpi-row", "children"),
    Output("main-graph", "figure"),
    Output("sample-size", "children"),
    Input("flt-contract", "value"),
    Input("flt-churn", "value"),
    Input("flt-tenure", "value"),
    Input("flt-charges", "value"),
    Input("chart-type", "value")
)
def update_dashboard(contract_sel, churn_sel, tenure_rng, charge_rng, chart_type):
    d = apply_filters(df, contract_sel, churn_sel, tenure_rng, charge_rng)  # df ya tiene 'Risk'

    churn_rate, avg_monthly, avg_tenure, _ = calc_kpis(d)
    avg_risk = d["Risk"].mean() * 100 if len(d) else 0.0

    kpis = [
        kpi_card("Churn Rate", f"{churn_rate:.1f}%", "#EF553B"),
        kpi_card("Avg Predicted Risk", f"{avg_risk:.1f}%", "#F59E0B"),
        kpi_card("Avg Monthly Charges", f"${avg_monthly:.2f}", "#10B981"),
        kpi_card("Avg Tenure", f"{avg_tenure:.1f} meses", "#636EFA"),
    ]

    # Gráfico principal
    if chart_type == "riskmap":
        fig = fig_riskmap(d if len(d) else d.iloc[0:0])
    elif chart_type == "riskcontract":
        fig = fig_risk_by_contract(d if len(d) else d.iloc[0:0])
    elif chart_type == "pie":
        fig = fig_pie(d if len(d) else df.iloc[0:0])
    elif chart_type == "contract":
        fig = fig_contract(d if len(d) else df.iloc[0:0])
    elif chart_type == "monthly":
        fig = fig_monthly(d if len(d) else df.iloc[0:0])
    else:
        fig = fig_box(d if len(d) else df.iloc[0:0])

    # Conteo detallado
    n_yes = int((d["Churn"]=="Yes").sum()) if len(d) else 0
    n_no  = int((d["Churn"]=="No").sum())  if len(d) else 0
    n_text = f"Muestra filtrada: {len(d):,} registros — Yes: {n_yes:,} | No: {n_no:,}"

    return kpis, fig, n_text

# ===== Scoring individual (gauge) vía API =====
@app.callback(
    Output("score-gauge", "figure"),
    Output("score-msg", "children"),
    Input("btn-score", "n_clicks"),
    State("sc-contract", "value"),
    State("sc-tenure", "value"),
    State("sc-charges", "value"),
    prevent_initial_call=True
)
def score_customer(n_clicks, ct, tn, ch):
    record = [{
        "Contract": ct,
        "Tenure": float(tn or 0),
        "MonthlyCharges": float(ch or 0)
    }]

    probs = call_api_batch(record)
    prob = probs[0] if probs else np.nan

    if np.isnan(prob):
        msg = "No se pudo obtener la predicción desde la API. Verifica que el endpoint /predict_batch esté activo."
        # Marcamos 0 solo para que el gauge pinte algo, pero el mensaje aclara que hubo error.
        return fig_gauge(0.0), msg

    msg = "Probabilidad estimada por el modelo desplegado (vía API)."
    return fig_gauge(prob), msg

# ===== Reset filtros =====
@app.callback(
    Output("flt-contract", "value"),
    Output("flt-churn", "value"),
    Output("flt-tenure", "value"),
    Output("flt-charges", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return (
        list(df["Contract"].unique()),
        ["Yes", "No"],
        [int(df["Tenure"].min()), int(df["Tenure"].max())],
        [float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())]
    )

# ===== Descargar CSV filtrado =====
@app.callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    State("flt-contract", "value"),
    State("flt-churn", "value"),
    State("flt-tenure", "value"),
    State("flt-charges", "value"),
    prevent_initial_call=True
)
def download_filtered(n_clicks, contract_sel, churn_sel, tenure_rng, charge_rng):
    d = apply_filters(df, contract_sel, churn_sel, tenure_rng, charge_rng)
    cols = [c for c in ["Churn","Risk","MonthlyCharges","Tenure","Contract"] if c in d.columns] + \
           [c for c in d.columns if c not in ["Churn","Risk","MonthlyCharges","Tenure","Contract"]]
    csv_string = d[cols].to_csv(index=False, encoding="utf-8")
    return dcc.send_bytes(lambda b: b.write(csv_string.encode("utf-8")), "telco_filtered.csv")


# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")   # <-- cambio importante
    print(f"\nDash corriendo en: http://{host}:{port}/\n")
    app.run(debug=True, host=host, port=port)

