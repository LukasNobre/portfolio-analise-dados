"""
============================================================
PROJETO 2: Análise de Tendência de Movimentação Portuária
           com Metodologia DEA (Data Envelopment Analysis)
============================================================
Descrição:
    Avaliação de eficiência operacional de terminais portuários
    usando DEA-CCR (retornos constantes de escala) e DEA-BCC
    (retornos variáveis). Identifica portos de referência (benchmarks),
    ineficientes e gera planos de melhoria operacional.

Competências demonstradas:
    - Modelagem de otimização linear (scipy.optimize)
    - Metodologia DEA (CCR e BCC)
    - Análise de séries temporais e sazonalidade
    - Identificação de benchmarks e gap analysis
    - Recomendações operacionais baseadas em dados

Referências:
    Charnes, Cooper & Rhodes (1978) — Modelo CCR
    Banker, Charnes & Cooper (1984) — Modelo BCC

Autor: [Seu Nome]
Data: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import linprog
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#F0F4F8",
    "axes.facecolor":   "#F0F4F8",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
OUTPUT_DIR = "output_dea_portuario"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DADOS DOS TERMINAIS PORTUÁRIOS
# ═══════════════════════════════════════════════════════════════════════════════
def criar_dataset_portuario() -> pd.DataFrame:
    """
    Dataset com 12 terminais portuários brasileiros.
    
    INPUTS (recursos utilizados):
        - Trabalhadores (número de funcionários)
        - Equipamentos (guindastes, reach stackers, etc.)
        - Área de pátio (hectares)
        - Custo operacional (R$ milhões/ano)
    
    OUTPUTS (resultados gerados):
        - TEU movimentados (Twenty-foot Equivalent Unit, mil/ano)
        - Receita bruta (R$ milhões/ano)
        - Taxa de ocupação berço (%)
        - NPS Armadores (satisfação, 0-100)
    """
    np.random.seed(42)

    dados = {
        "Terminal": [
            "Santos - Terminal A", "Santos - Terminal B",
            "Paranaguá - TGG",    "Rio de Janeiro - MultiRio",
            "Itajaí - Portonave", "Suape - TECON",
            "Pecém - CIPP",       "Vitória - TVV",
            "Rio Grande - TECON", "Manaus - CIMPORT",
            "Fortaleza - ETC",    "Salvador - TMFS",
        ],
        "UF": ["SP","SP","PR","RJ","SC","PE","CE","ES","RS","AM","CE","BA"],

        # Inputs
        "Trabalhadores":   [1850,1420,980,1100,850,720,640,780,910,540,480,620],
        "Equipamentos":    [42,  35,  22, 28,  19, 18, 14, 20, 24, 12, 11, 16],
        "Area_Patio_ha":   [120, 95,  58, 70,  48, 42, 36, 52, 62, 28, 25, 40],
        "Custo_Op_MM":     [280, 210, 145,170, 118,102, 88,128,148, 72, 68, 95],

        # Outputs
        "TEU_mil":         [1850,1380, 980,1050, 920, 680,510, 740, 820,380,310,510],
        "Receita_MM":      [520, 385,  270, 295, 258, 190,148, 210, 235,108, 92,145],
        "Ocup_Berco_pct":  [87,  82,   78,  80,  85,  72, 68,  74,  78, 65, 62, 70],
        "NPS_Armadores":   [78,  71,   80,  68,  88,  73, 65,  70,  75, 60, 58, 66],
    }

    df = pd.DataFrame(dados)

    # Adiciona séries temporais mensais (2020-2023)
    meses = pd.date_range("2020-01", periods=48, freq="MS")
    series = {}
    for _, row in df.iterrows():
        base = row["TEU_mil"] / 12
        tendencia = np.linspace(0.85, 1.10, 48)
        sazonal   = 1 + 0.12 * np.sin(np.linspace(0, 6*np.pi, 48))
        ruido     = np.random.normal(1, 0.04, 48)
        serie     = base * tendencia * sazonal * ruido
        series[row["Terminal"]] = np.maximum(serie, 0)

    df_series = pd.DataFrame(series, index=meses)
    df_series.index.name = "Data"

    return df, df_series


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODELO DEA — CCR (Charnes, Cooper & Rhodes, 1978)
# ═══════════════════════════════════════════════════════════════════════════════
def dea_ccr(inputs: np.ndarray, outputs: np.ndarray) -> dict:
    """
    Implementa o modelo DEA-CCR orientado a inputs (minimização de recursos).
    
    Formulação:
        max  θ = u'y_0
        s.t. v'x_0 = 1
             u'Y - v'X ≤ 0
             u, v ≥ 0
    
    Convertido para programação linear via dualidade de Farrell.
    
    Args:
        inputs:  matriz (n_DMU × n_inputs)
        outputs: matriz (n_DMU × n_outputs)
    
    Returns:
        dict com escores de eficiência, folgas e pesos.
    """
    n_dmu, n_inp = inputs.shape
    _, n_out     = outputs.shape
    escores      = np.zeros(n_dmu)
    lambdas_all  = np.zeros((n_dmu, n_dmu))
    folgas_inp   = np.zeros((n_dmu, n_inp))
    folgas_out   = np.zeros((n_dmu, n_out))

    for i in range(n_dmu):
        # Problema dual (forma envolvente — envelope form)
        # min θ
        # s.t. X·λ ≤ θ·x_0   (inputs da DMU i)
        #      Y·λ ≥ y_0       (outputs da DMU i)
        #      λ ≥ 0

        # Variáveis: [θ, λ_1, ..., λ_n]  → n+1 variáveis
        c = np.zeros(1 + n_dmu)
        c[0] = 1.0  # minimizar θ

        # Restrições de input: X·λ - θ·x_0 ≤ 0
        # → para cada input k: sum_j(X[j,k]*λ_j) - x_0[k]*θ ≤ 0
        A_ub = np.zeros((n_inp + n_out, 1 + n_dmu))
        b_ub = np.zeros(n_inp + n_out)

        for k in range(n_inp):
            A_ub[k, 0] = -inputs[i, k]      # coef de θ
            A_ub[k, 1:] = inputs[:, k]       # coef de λ

        # Restrições de output: -Y·λ ≤ -y_0
        for r in range(n_out):
            A_ub[n_inp + r, 1:] = -outputs[:, r]
            b_ub[n_inp + r]     = -outputs[i, r]

        bounds = [(0, None)] * (1 + n_dmu)
        bounds[0] = (0, 1.0)  # 0 ≤ θ ≤ 1

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")

        if res.success:
            theta    = res.x[0]
            lam      = res.x[1:]
            escores[i]       = min(theta, 1.0)
            lambdas_all[i]   = lam

            # Calcular folgas
            proj_inp = inputs[i] * theta
            bench_inp = inputs.T @ lam
            folgas_inp[i] = proj_inp - bench_inp

            bench_out = outputs.T @ lam
            folgas_out[i] = np.maximum(0, outputs[i] - bench_out)
        else:
            escores[i] = np.nan

    return {
        "escores":     escores,
        "lambdas":     lambdas_all,
        "folgas_inp":  folgas_inp,
        "folgas_out":  folgas_out,
    }


def dea_bcc(inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Modelo DEA-BCC (retornos variáveis de escala).
    Adiciona restrição convexidade: sum(λ) = 1.
    """
    n_dmu, n_inp = inputs.shape
    _, n_out     = outputs.shape
    escores      = np.zeros(n_dmu)

    for i in range(n_dmu):
        c = np.zeros(1 + n_dmu)
        c[0] = 1.0

        A_ub = np.zeros((n_inp + n_out, 1 + n_dmu))
        b_ub = np.zeros(n_inp + n_out)

        for k in range(n_inp):
            A_ub[k, 0] = -inputs[i, k]
            A_ub[k, 1:] = inputs[:, k]
        for r in range(n_out):
            A_ub[n_inp + r, 1:] = -outputs[:, r]
            b_ub[n_inp + r]     = -outputs[i, r]

        # Convexidade: sum(λ) = 1
        A_eq = np.zeros((1, 1 + n_dmu))
        A_eq[0, 1:] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * (1 + n_dmu)
        bounds[0] = (0, 1.0)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        escores[i] = min(res.x[0], 1.0) if res.success else np.nan

    return escores


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANÁLISE DE TENDÊNCIA DAS SÉRIES TEMPORAIS
# ═══════════════════════════════════════════════════════════════════════════════
def analisar_tendencias(df_series: pd.DataFrame) -> pd.DataFrame:
    """Regressão linear por terminal para extração de tendência."""
    resultados = []
    t = np.arange(len(df_series))

    for col in df_series.columns:
        y = df_series[col].values
        slope, intercept, r_value, p_value, se = stats.linregress(t, y)
        resultados.append({
            "Terminal":           col,
            "Tendencia_mensal":   slope,
            "Tendencia_anual":    slope * 12,
            "R2":                 r_value**2,
            "p_valor":            p_value,
            "Significativa":      "Sim" if p_value < 0.05 else "Não",
            "Crescimento_total":  (y[-1] - y[0]) / y[0] * 100,
        })

    return pd.DataFrame(resultados).sort_values("Tendencia_anual", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLANO DE MELHORIA OPERACIONAL
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_plano_melhoria(df: pd.DataFrame, escores_ccr: np.ndarray,
                          folgas_inp: np.ndarray, folgas_out: np.ndarray,
                          col_inp: list, col_out: list) -> pd.DataFrame:
    """
    Para cada DMU ineficiente, calcula reduções necessárias nos inputs
    e aumentos nos outputs para atingir a fronteira de eficiência.
    """
    df_res = df[["Terminal","UF"]].copy()
    df_res["Escore_CCR"] = np.round(escores_ccr * 100, 2)
    df_res["Status"]     = np.where(escores_ccr >= 0.999, "✅ Eficiente",
                           np.where(escores_ccr >= 0.85, "⚠️ Levemente Inef.",
                                    "❌ Ineficiente"))

    # Reduções de input necessárias (%)
    for j, col in enumerate(col_inp):
        df_res[f"Reduzir_{col}_pct"] = np.round(
            folgas_inp[:, j] / df[col].values * 100, 1
        )

    # Aumentos de output necessários (%)
    for r, col in enumerate(col_out):
        df_res[f"Aumentar_{col}_pct"] = np.round(
            folgas_out[:, r] / df[col].values * 100, 1
        )

    return df_res.sort_values("Escore_CCR", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════
def visualizar_dea(df: pd.DataFrame, df_res: pd.DataFrame,
                   df_series: pd.DataFrame, df_tend: pd.DataFrame,
                   escores_bcc: np.ndarray):
    """Painel completo DEA + Tendências."""
    cores_status = {"✅ Eficiente": "#2E7D32",
                    "⚠️ Levemente Inef.": "#F9A825",
                    "❌ Ineficiente":     "#C62828"}

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle("Análise DEA — Eficiência de Terminais Portuários",
                 fontsize=18, fontweight="bold", color="#0D47A1", y=0.98)

    terminais_curtos = [t.split(" - ")[1] if " - " in t else t.split()[0]
                        for t in df_res["Terminal"]]

    # --- Plot 1: Escores CCR ---
    ax1 = fig.add_subplot(gs[0, 0])
    cores = [cores_status[s] for s in df_res["Status"]]
    bars  = ax1.barh(terminais_curtos, df_res["Escore_CCR"],
                     color=cores, edgecolor="white", height=0.65)
    ax1.axvline(100, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.set_xlim(50, 108)
    ax1.set_title("Escores DEA-CCR (%)", fontweight="bold", color="#0D47A1")
    ax1.set_xlabel("Eficiência (%)")
    for bar, val in zip(bars, df_res["Escore_CCR"]):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    # --- Plot 2: CCR vs BCC ---
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(terminais_curtos))
    w = 0.35
    ax2.bar(x - w/2, df_res["Escore_CCR"], w, label="CCR", color="#1565C0", alpha=0.85)
    ax2.bar(x + w/2, escores_bcc * 100,    w, label="BCC", color="#42A5F5", alpha=0.85)
    ax2.axhline(100, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xticks(x); ax2.set_xticklabels(terminais_curtos, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Eficiência (%)"); ax2.set_ylim(50, 110)
    ax2.set_title("CCR vs BCC por Terminal", fontweight="bold", color="#0D47A1")
    ax2.legend(fontsize=9)

    # --- Plot 3: Escala de eficiência (CCR/BCC) ---
    ax3 = fig.add_subplot(gs[0, 2])
    escala = df_res["Escore_CCR"].values / (escores_bcc * 100 + 1e-9)
    cor_esc = ["#C62828" if e < 0.95 else "#2E7D32" for e in escala]
    ax3.bar(terminais_curtos, escala, color=cor_esc, edgecolor="white", alpha=0.85)
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax3.set_xticklabels(terminais_curtos, rotation=45, ha="right", fontsize=8)
    ax3.set_title("Eficiência de Escala (CCR/BCC)", fontweight="bold", color="#0D47A1")
    ax3.set_ylabel("Razão de Escala")

    # --- Plot 4: Tendência dos Top-5 terminais ---
    ax4 = fig.add_subplot(gs[1, :2])
    top5 = df_tend.head(5)["Terminal"].tolist()
    paleta = ["#1565C0","#388E3C","#F9A825","#C62828","#7B1FA2"]
    for col, cor in zip(top5, paleta):
        ax4.plot(df_series.index, df_series[col],
                 label=col.split(" - ")[-1], color=cor, linewidth=1.8, alpha=0.9)
    ax4.set_title("Tendência de Movimentação — Top 5 Terminais (2020-2023)",
                  fontweight="bold", color="#0D47A1")
    ax4.set_xlabel("Data"); ax4.set_ylabel("TEU (mil/mês)")
    ax4.legend(fontsize=8, ncol=2); ax4.xaxis.set_major_locator(plt.MaxNLocator(8))

    # --- Plot 5: Gap de melhoria ---
    ax5 = fig.add_subplot(gs[1, 2])
    inef = df_res[df_res["Escore_CCR"] < 99.9].head(6)
    gap  = 100 - inef["Escore_CCR"].values
    nomes = [t.split(" - ")[-1] for t in inef["Terminal"].tolist()]
    ax5.barh(nomes, gap, color="#C62828", edgecolor="white", alpha=0.85)
    ax5.set_title("Gap de Eficiência (%)\nTerminais Ineficientes",
                  fontweight="bold", color="#0D47A1")
    ax5.set_xlabel("Gap para Fronteira (%)")
    for i, g in enumerate(gap):
        ax5.text(g + 0.1, i, f"{g:.1f}%", va="center", fontsize=9, color="#C62828")

    plt.savefig(f"{OUTPUT_DIR}/dea_portuario_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#F0F4F8")
    print(f"✅ Dashboard DEA salvo em: {OUTPUT_DIR}/dea_portuario_dashboard.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PROJETO 2 — DEA: Eficiência de Terminais Portuários")
    print("=" * 65)

    print("\n[1/5] Carregando dados dos terminais...")
    df, df_series = criar_dataset_portuario()
    print(f"  {len(df)} terminais | {len(df_series)} meses de série histórica")

    col_inp = ["Trabalhadores","Equipamentos","Area_Patio_ha","Custo_Op_MM"]
    col_out = ["TEU_mil","Receita_MM","Ocup_Berco_pct","NPS_Armadores"]

    X = df[col_inp].values.astype(float)
    Y = df[col_out].values.astype(float)

    print("\n[2/5] Executando modelo DEA-CCR...")
    res_ccr = dea_ccr(X, Y)
    print("  Modelo CCR concluído.")

    print("\n[3/5] Executando modelo DEA-BCC...")
    escores_bcc = dea_bcc(X, Y)
    print("  Modelo BCC concluído.")

    print("\n[4/5] Analisando tendências temporais...")
    df_tend = analisar_tendencias(df_series)

    df_res = gerar_plano_melhoria(df, res_ccr["escores"],
                                   res_ccr["folgas_inp"], res_ccr["folgas_out"],
                                   col_inp, col_out)

    # Relatório textual
    print("\n" + "─"*65)
    print(f"{'RANKING DE EFICIÊNCIA — MODELO DEA-CCR':^65}")
    print("─"*65)
    print(f"{'Terminal':<28} {'CCR':>7} {'BCC':>7} {'Status':<22}")
    print("─"*65)
    for i, row in df_res.iterrows():
        bcc = escores_bcc[df.index.get_loc(i)] * 100
        print(f"  {row['Terminal']:<26} {row['Escore_CCR']:>6.1f}% "
              f"{bcc:>6.1f}%  {row['Status']}")

    n_ef = (res_ccr["escores"] >= 0.999).sum()
    print("─"*65)
    print(f"  Terminais eficientes (score=100%): {n_ef}/{len(df)}")
    print(f"  Eficiência média: {res_ccr['escores'].mean()*100:.1f}%")

    print("\n📈 Top 3 Tendências de Crescimento:")
    for _, r in df_tend.head(3).iterrows():
        print(f"  {r['Terminal']:<30} +{r['Tendencia_anual']:.1f} TEU mil/ano  "
              f"(R²={r['R2']:.3f})")

    print("\n[5/5] Gerando visualizações...")
    visualizar_dea(df, df_res, df_series, df_tend, escores_bcc)

    df_res.to_csv(f"{OUTPUT_DIR}/resultados_dea.csv",
                  sep=";", decimal=",", encoding="utf-8-sig", index=False)
    df_tend.to_csv(f"{OUTPUT_DIR}/tendencias_portuarias.csv",
                   sep=";", decimal=",", encoding="utf-8-sig", index=False)

    print("\n" + "=" * 65)
    print("  ✅ PROJETO CONCLUÍDO COM SUCESSO!")
    print(f"  📁 Resultados em: ./{OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
