"""
============================================================
PROJETO 1: Análise de Dados com Exportação para Power BI
============================================================
Descrição:
    Pipeline completo de análise de dados de vendas com limpeza,
    transformação, KPIs e exportação de datasets otimizados
    para consumo no Power BI via arquivos .csv e .xlsx.

Competências demonstradas:
    - ETL com Pandas
    - Análise Exploratória (EDA)
    - Geração de KPIs e métricas de negócio
    - Visualizações estáticas com Matplotlib/Seaborn
    - Exportação de dados estruturados para Power BI

Autor: [Seu Nome]
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import datetime, timedelta
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Configurações Visuais ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams.update({
    "figure.facecolor": "#F8FAFC",
    "axes.facecolor":   "#F8FAFC",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
OUTPUT_DIR = "output_powerbi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GERAÇÃO DE DADOS SINTÉTICOS (simula carga de banco / ERP)
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_dados_vendas(seed: int = 42, n: int = 5_000) -> pd.DataFrame:
    """Gera dataset sintético realista de vendas para análise."""
    random.seed(seed)
    np.random.seed(seed)

    regioes      = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    categorias   = ["Eletrônicos", "Vestuário", "Alimentos", "Móveis", "Saúde"]
    canais       = ["E-commerce", "Loja Física", "Televendas", "Marketplace"]
    vendedores   = [f"Vendedor_{i:02d}" for i in range(1, 21)]

    peso_regiao  = [0.10, 0.20, 0.15, 0.35, 0.20]
    peso_canal   = [0.40, 0.30, 0.10, 0.20]
    preco_base   = {"Eletrônicos": 1200, "Vestuário": 180, "Alimentos": 55,
                    "Móveis": 950, "Saúde": 210}

    datas = [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 364))
             for _ in range(n)]

    df = pd.DataFrame({
        "data_venda":  datas,
        "regiao":      np.random.choice(regioes,    n, p=peso_regiao),
        "categoria":   np.random.choice(categorias, n),
        "canal":       np.random.choice(canais,     n, p=peso_canal),
        "vendedor":    np.random.choice(vendedores, n),
        "quantidade":  np.random.randint(1, 15, n),
    })

    df["preco_unitario"] = df["categoria"].map(preco_base) * (
        1 + np.random.normal(0, 0.15, n)
    )
    df["desconto_pct"]   = np.clip(np.random.exponential(0.08, n), 0, 0.35)
    df["custo_unitario"] = df["preco_unitario"] * np.random.uniform(0.45, 0.65, n)
    df["frete"]          = np.random.uniform(8, 60, n) * df["quantidade"]

    df["receita_bruta"]  = df["preco_unitario"] * df["quantidade"]
    df["desconto_R$"]    = df["receita_bruta"] * df["desconto_pct"]
    df["receita_liquida"]= df["receita_bruta"] - df["desconto_R$"]
    df["custo_total"]    = df["custo_unitario"] * df["quantidade"] + df["frete"]
    df["lucro"]          = df["receita_liquida"] - df["custo_total"]
    df["margem_pct"]     = df["lucro"] / df["receita_liquida"] * 100

    return df.sort_values("data_venda").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LIMPEZA & TRANSFORMAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
def limpar_transformar(df: pd.DataFrame) -> pd.DataFrame:
    """ETL: limpeza e enriquecimento temporal."""
    print("📋 Shape original:", df.shape)

    # Remove duplicatas
    antes = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicatas removidas: {antes - len(df)}")

    # Remove outliers extremos (preço negativo ou lucro absurdo)
    df = df[df["preco_unitario"] > 0]
    df = df[df["lucro"].between(df["lucro"].quantile(0.005),
                                df["lucro"].quantile(0.995))]

    # Enriquecimento temporal
    df["data_venda"]  = pd.to_datetime(df["data_venda"])
    df["ano"]         = df["data_venda"].dt.year
    df["mes"]         = df["data_venda"].dt.month
    df["mes_nome"]    = df["data_venda"].dt.strftime("%b")
    df["trimestre"]   = df["data_venda"].dt.quarter.map(
                            {1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    df["dia_semana"]  = df["data_venda"].dt.day_name()
    df["semana_ano"]  = df["data_venda"].dt.isocalendar().week.astype(int)

    # Segmentação de margem
    df["faixa_margem"] = pd.cut(
        df["margem_pct"],
        bins=[-np.inf, 0, 10, 25, np.inf],
        labels=["Prejuízo", "Baixa", "Média", "Alta"]
    )

    print("📋 Shape após limpeza:", df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KPIs
# ═══════════════════════════════════════════════════════════════════════════════
def calcular_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs executivos de negócio."""
    kpis = {
        "Receita Bruta Total (R$)":    df["receita_bruta"].sum(),
        "Receita Líquida Total (R$)":  df["receita_liquida"].sum(),
        "Lucro Total (R$)":            df["lucro"].sum(),
        "Margem Média (%)":            df["margem_pct"].mean(),
        "Ticket Médio (R$)":           df["receita_liquida"].mean(),
        "Total de Pedidos":            len(df),
        "Desconto Médio (%)":          df["desconto_pct"].mean() * 100,
        "Receita por Pedido (R$)":     df["receita_liquida"].sum() / len(df),
    }

    print("\n📊 KPIs Principais:")
    print("─" * 45)
    for k, v in kpis.items():
        if "%" in k:
            print(f"  {k:<30} {v:>8.2f}%")
        elif "Total de" in k:
            print(f"  {k:<30} {v:>8,.0f}")
        else:
            print(f"  {k:<30} R$ {v:>10,.2f}")
    print("─" * 45)
    return kpis


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_visualizacoes(df: pd.DataFrame):
    """Gera painel visual com 6 gráficos analíticos."""
    cores = ["#1565C0","#1976D2","#42A5F5","#90CAF9","#BBDEFB"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Dashboard Analítico de Vendas — 2023",
                 fontsize=18, fontweight="bold", y=1.01, color="#1A237E")
    fig.tight_layout(pad=4)

    # --- Gráfico 1: Receita mensal ---
    ax = axes[0, 0]
    mensal = df.groupby("mes")["receita_liquida"].sum() / 1e6
    mensal.plot(kind="bar", ax=ax, color=cores[0], edgecolor="white", width=0.7)
    ax.set_title("Receita Líquida Mensal", fontweight="bold")
    ax.set_xlabel("Mês"); ax.set_ylabel("R$ Milhões")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"R${x:.1f}M"))
    ax.set_xticklabels(range(1, 13), rotation=0)

    # --- Gráfico 2: Receita por categoria ---
    ax = axes[0, 1]
    cat = df.groupby("categoria")["receita_liquida"].sum().sort_values()
    cat.plot(kind="barh", ax=ax, color=cores, edgecolor="white")
    ax.set_title("Receita por Categoria", fontweight="bold")
    ax.set_xlabel("R$")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))

    # --- Gráfico 3: Margem por canal ---
    ax = axes[0, 2]
    canal_margem = df.groupby("canal")["margem_pct"].mean().sort_values()
    bars = canal_margem.plot(kind="barh", ax=ax, color="#42A5F5", edgecolor="white")
    ax.set_title("Margem Média por Canal", fontweight="bold")
    ax.set_xlabel("Margem (%)")
    for i, v in enumerate(canal_margem):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10, color="#1565C0")

    # --- Gráfico 4: Evolução semanal (linha) ---
    ax = axes[1, 0]
    semanal = df.groupby("semana_ano")["receita_liquida"].sum() / 1e3
    ax.plot(semanal.index, semanal.values, color=cores[0], linewidth=2.5)
    ax.fill_between(semanal.index, semanal.values, alpha=0.15, color=cores[0])
    ax.set_title("Evolução Semanal de Receita", fontweight="bold")
    ax.set_xlabel("Semana do Ano"); ax.set_ylabel("R$ Mil")

    # --- Gráfico 5: Distribuição margem ---
    ax = axes[1, 1]
    df["margem_pct"].plot(kind="hist", bins=40, ax=ax,
                           color=cores[0], edgecolor="white", alpha=0.85)
    ax.axvline(df["margem_pct"].mean(), color="red", linestyle="--",
               linewidth=2, label=f"Média: {df['margem_pct'].mean():.1f}%")
    ax.set_title("Distribuição da Margem (%)", fontweight="bold")
    ax.set_xlabel("Margem (%)"); ax.legend()

    # --- Gráfico 6: Receita por região ---
    ax = axes[1, 2]
    regiao = df.groupby("regiao")["receita_liquida"].sum().sort_values(ascending=False)
    wedges, texts, autotexts = ax.pie(
        regiao, labels=regiao.index, autopct="%1.1f%%",
        colors=cores, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax.set_title("Receita por Região", fontweight="bold")

    plt.savefig(f"{OUTPUT_DIR}/dashboard_vendas.png",
                dpi=150, bbox_inches="tight", facecolor="#F8FAFC")
    print(f"\n✅ Dashboard salvo em: {OUTPUT_DIR}/dashboard_vendas.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EXPORTAÇÃO PARA POWER BI
# ═══════════════════════════════════════════════════════════════════════════════
def exportar_para_powerbi(df: pd.DataFrame, kpis: dict):
    """
    Exporta tabelas dimensionais e fato para uso no Power BI.
    Modelo estrela: fato_vendas + dim_tempo + dim_produto + dim_geografia.
    """
    print("\n📤 Exportando datasets para Power BI...")

    # Tabela Fato
    fato = df[["data_venda","regiao","categoria","canal","vendedor",
               "quantidade","receita_bruta","desconto_R$","receita_liquida",
               "custo_total","lucro","margem_pct"]].copy()
    fato.to_csv(f"{OUTPUT_DIR}/fato_vendas.csv", index=False,
                sep=";", decimal=",", encoding="utf-8-sig")

    # Dimensão Tempo
    dim_tempo = df[["data_venda","ano","mes","mes_nome",
                    "trimestre","dia_semana","semana_ano"]].drop_duplicates()
    dim_tempo.to_csv(f"{OUTPUT_DIR}/dim_tempo.csv", index=False,
                     sep=";", decimal=",", encoding="utf-8-sig")

    # Dimensão Geografia
    dim_geo = df[["regiao"]].drop_duplicates().reset_index(drop=True)
    dim_geo["id_regiao"] = dim_geo.index + 1
    dim_geo.to_csv(f"{OUTPUT_DIR}/dim_geografia.csv", index=False,
                   sep=";", decimal=",", encoding="utf-8-sig")

    # KPIs resumidos
    df_kpis = pd.DataFrame(list(kpis.items()), columns=["KPI","Valor"])
    df_kpis.to_csv(f"{OUTPUT_DIR}/kpis_executivo.csv", index=False,
                   sep=";", decimal=",", encoding="utf-8-sig")

    # Resumo mensal (tabela pronta para cartão no Power BI)
    mensal = df.groupby(["ano","trimestre","mes","mes_nome"]).agg(
        Receita_Liquida=("receita_liquida","sum"),
        Lucro=("lucro","sum"),
        Pedidos=("lucro","count"),
        Margem_Media=("margem_pct","mean"),
    ).reset_index()
    mensal.to_csv(f"{OUTPUT_DIR}/resumo_mensal.csv", index=False,
                  sep=";", decimal=",", encoding="utf-8-sig")

    arquivos = ["fato_vendas.csv","dim_tempo.csv","dim_geografia.csv",
                "kpis_executivo.csv","resumo_mensal.csv"]
    for arq in arquivos:
        caminho = f"{OUTPUT_DIR}/{arq}"
        tamanho = os.path.getsize(caminho) / 1024
        print(f"  ✔ {arq:<35} {tamanho:>7.1f} KB")
    print("\n💡 Dica: Importe esses CSVs no Power BI → Obter Dados → Texto/CSV")
    print("         Use ponto-e-vírgula como delimitador e vírgula como decimal.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PROJETO 1 — Análise de Dados + Exportação Power BI")
    print("=" * 60)

    print("\n[1/5] Gerando dados sintéticos de vendas...")
    df = gerar_dados_vendas()

    print("\n[2/5] Limpeza e transformação (ETL)...")
    df = limpar_transformar(df)

    print("\n[3/5] Calculando KPIs de negócio...")
    kpis = calcular_kpis(df)

    print("\n[4/5] Gerando visualizações...")
    gerar_visualizacoes(df)

    print("\n[5/5] Exportando para Power BI...")
    exportar_para_powerbi(df, kpis)

    print("\n" + "=" * 60)
    print("  ✅ PROJETO CONCLUÍDO COM SUCESSO!")
    print(f"  📁 Arquivos gerados em: ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
