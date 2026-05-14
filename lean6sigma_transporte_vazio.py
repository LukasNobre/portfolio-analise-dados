"""
============================================================
PROJETO 3: Redução do Custo de Transporte Vazio
           com Metodologia Lean Six Sigma (DMAIC)
============================================================
Descrição:
    Projeto completo Lean Six Sigma seguindo o ciclo DMAIC para
    identificar, medir, analisar, melhorar e controlar os custos
    de viagens vazias em uma transportadora de carga.

    Fases DMAIC implementadas:
        D — Define:   Mapeamento do processo e definição do problema
        M — Measure:  KPIs, baseline e estabilidade do processo
        A — Analyze:  Causa-raiz (Ishikawa, Pareto, correlações)
        I — Improve:  Cenários de melhoria e ROI
        C — Control:  Plano de controle com limites estatísticos (CEP)

Competências demonstradas:
    - Metodologia Lean Six Sigma (DMAIC)
    - CEP — Controle Estatístico do Processo (Cartas X-bar e R)
    - Análise de Pareto (Princípio 80/20)
    - DPMO e nível Sigma
    - Simulação de Monte Carlo para ROI
    - Regressão para identificação de drivers de custo

Autor: [Seu Nome]
Data: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
OUTPUT_DIR = "output_lean6sigma"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Paleta Lean ──────────────────────────────────────────────────────────────
VERDE   = "#2E7D32"
AMARELO = "#F9A825"
VERMELHO= "#C62828"
AZUL    = "#1565C0"
CINZA   = "#616161"


# ═══════════════════════════════════════════════════════════════════════════════
# FASE D — DEFINE
# ═══════════════════════════════════════════════════════════════════════════════
def fase_define():
    """
    Imprime o Project Charter e o mapa de processo (SIPOC simplificado).
    """
    print("\n" + "═"*65)
    print("  FASE D — DEFINE")
    print("═"*65)
    print("""
  📋 PROJECT CHARTER
  ─────────────────────────────────────────────────────────────
  Problema:    Viagens vazias representam 38% das km rodadas,
               gerando custo médio de R$ 1,85 M/mês desnecessários.
               Meta: Reduzir para ≤ 22% em 6 meses.

  CTQ (Critical-to-Quality):
    ► Taxa de Viagem Vazia (TVV) — meta: ≤ 22%
    ► Custo por km vazio — meta: ≤ R$ 3,20/km
    ► OTIF (On-Time In Full) — mínimo: 94%

  Escopo:      Rotas regionais (SP, PR, SC, RS) — frota própria
  Equipe:      Black Belt + 2 Green Belts + Ops + TI + Comercial
  Prazo:       6 meses | Início: Janeiro/2024
  Retorno Est: R$ 1,2 M/ano em economia direta
  ─────────────────────────────────────────────────────────────

  🗺️  SIPOC — Mapa de Alto Nível
  ─────────────────────────────────────────────────────────────
  Suppliers  → Clientes, Operadores de pátio, Manutenção
  Inputs     → Pedidos de coleta, disponibilidade de frota,
               histórico de rotas
  Process    → Planejamento → Despacho → Execução → Retorno
  Outputs    → Entrega realizada, km rodados, custo por viagem
  Customers  → Clientes B2B, Financeiro, Operações
  ─────────────────────────────────────────────────────────────
""")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE M — MEASURE (Geração de dados + KPIs)
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_dados(seed: int = 42) -> pd.DataFrame:
    """Gera 18 meses de dados operacionais da frota."""
    np.random.seed(seed)

    datas  = pd.date_range("2023-01-01", periods=540, freq="D")  # ~18 meses
    rotas  = ["SP→PR", "SP→SC", "SP→RS", "PR→SC", "PR→RS", "SC→RS",
              "SP→MG", "PR→SP", "SC→SP", "RS→SP"]
    veiculos = [f"CAM-{i:03d}" for i in range(1, 51)]  # 50 caminhões

    n = len(datas) * 3
    dados = {
        "data":         np.random.choice(datas, n),
        "rota":         np.random.choice(rotas, n, p=[.22,.18,.15,.12,.10,.08,.05,.04,.03,.03]),
        "veiculo":      np.random.choice(veiculos, n),
        "km_total":     np.random.normal(480, 95, n).clip(150, 900),
        "carga_ton":    np.random.exponential(18, n).clip(0, 28),
        "motorista_exp_anos": np.random.randint(1, 20, n),
    }

    df = pd.DataFrame(dados)
    df["data"] = pd.to_datetime(df["data"])

    # Viagem vazia: carga < 2 ton → vazia
    df["viagem_vazia"] = (df["carga_ton"] < 2.0).astype(int)
    df["km_vazio"]     = df["km_total"] * df["viagem_vazia"]

    # Custo por km (vazio mais caro proporcionalmente)
    custo_km_base = np.random.normal(3.45, 0.40, n).clip(2.5, 5.5)
    df["custo_km"]     = custo_km_base * (1 + 0.12 * df["viagem_vazia"])
    df["custo_total"]  = df["km_total"] * df["custo_km"]
    df["custo_vazio"]  = df["km_vazio"] * df["custo_km"]

    # Sazonalidade e tendência
    mes_num = df["data"].dt.month
    df["custo_km"] *= (1 + 0.05 * np.sin(mes_num * np.pi / 6))

    df["mes"]       = df["data"].dt.to_period("M")
    df["semana"]    = df["data"].dt.isocalendar().week.astype(int)
    df["dia_semana"]= df["data"].dt.day_name()

    return df.sort_values("data").reset_index(drop=True)


def fase_measure(df: pd.DataFrame) -> dict:
    """Calcula baseline e KPIs do processo."""
    print("\n" + "═"*65)
    print("  FASE M — MEASURE")
    print("═"*65)

    tvv   = df["viagem_vazia"].mean() * 100
    custo = df["custo_vazio"].sum() / 1e6
    km_v  = df["km_vazio"].mean()
    cpkm  = df[df["viagem_vazia"]==1]["custo_km"].mean()

    # Nível Sigma
    oportunidades = len(df)
    defeitos       = df["viagem_vazia"].sum()
    dpmo           = defeitos / oportunidades * 1_000_000
    # Conversão DPMO → Sigma (aproximação)
    nivel_sigma    = norm.ppf(1 - dpmo/1_000_000) + 1.5

    kpis = {
        "Taxa Viagem Vazia (%)":          tvv,
        "Custo Total Vazio (R$ M)":        custo,
        "Km Médio por Viagem Vazia":       km_v,
        "Custo/km Vazio (R$)":             cpkm,
        "Total de Viagens":                len(df),
        "Viagens Vazias":                  defeitos,
        "DPMO":                            dpmo,
        "Nível Sigma":                     max(nivel_sigma, 0),
    }

    print("\n  📊 Baseline — KPIs Medidos (pré-melhoria):")
    print("  " + "─"*55)
    for k, v in kpis.items():
        if "%" in k:     print(f"    {k:<38} {v:>7.1f}%")
        elif "Sigma" in k: print(f"    {k:<38} {v:>7.2f}σ")
        elif "DPMO" in k:  print(f"    {k:<38} {v:>7,.0f}")
        elif "Total" in k and "R$" not in k: print(f"    {k:<38} {v:>7,.0f}")
        elif "R$" in k:  print(f"    {k:<38} R$ {v:>5.2f}")
        else:            print(f"    {k:<38} {v:>7,.1f}")
    print("  " + "─"*55)
    print(f"\n  ⚠️  Processo opera em {kpis['Nível Sigma']:.2f}σ — "
          f"meta: ≥ 4.0σ (TVV ≤ 22%)\n")
    return kpis


# ═══════════════════════════════════════════════════════════════════════════════
# FASE A — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
def fase_analyze(df: pd.DataFrame):
    """Análise de causa-raiz: Pareto, correlações e regressão."""
    print("═"*65)
    print("  FASE A — ANALYZE")
    print("═"*65)

    # Pareto das causas (simulado com categorias reais)
    causas = {
        "Falta de carga de retorno":              34.2,
        "Planejamento ineficiente de rotas":      22.8,
        "Cancelamento de pedidos last-minute":    14.1,
        "Clientes sem contraentrega":             11.3,
        "Desequilíbrio regional de demanda":       8.7,
        "Falha na comunicação comercial/ops":      5.4,
        "Outros":                                  3.5,
    }
    df_pareto = pd.DataFrame(list(causas.items()),
                              columns=["Causa","Frequência_%"])
    df_pareto = df_pareto.sort_values("Frequência_%", ascending=False)
    df_pareto["Acumulado_%"] = df_pareto["Frequência_%"].cumsum()

    print("\n  📌 Diagrama de Pareto — Causas Principais:")
    print("  " + "─"*55)
    for _, r in df_pareto.iterrows():
        barra = "█" * int(r["Frequência_%"] / 2)
        print(f"    {r['Causa']:<42} {r['Frequência_%']:>5.1f}%  {barra}")
    print(f"\n  ✔ 80/20: Top 3 causas = "
          f"{df_pareto.head(3)['Frequência_%'].sum():.1f}% dos defeitos\n")

    # Correlação: experiência do motorista vs viagem vazia
    corr, pval = stats.pointbiserialr(df["motorista_exp_anos"],
                                       df["viagem_vazia"])
    print(f"  📉 Correlação experiência motorista ↔ viagem vazia: "
          f"r={corr:.3f} (p={pval:.4f})")

    # Regressão: custo por km ~ carga + km_total
    X = df[["carga_ton","km_total","motorista_exp_anos"]]
    y = df["custo_km"]
    corrs = X.corrwith(y)
    print("\n  📐 Correlações com Custo/km:")
    for col, val in corrs.items():
        print(f"    {col:<30} r = {val:+.3f}")

    return df_pareto


# ═══════════════════════════════════════════════════════════════════════════════
# FASE I — IMPROVE
# ═══════════════════════════════════════════════════════════════════════════════
def fase_improve(kpis: dict) -> pd.DataFrame:
    """Simula cenários de melhoria e calcula ROI via Monte Carlo."""
    print("\n" + "═"*65)
    print("  FASE I — IMPROVE")
    print("═"*65)

    cenarios = pd.DataFrame({
        "Ação": [
            "Plataforma de carga de retorno (marketplace)",
            "Otimização algorítmica de rotas (VRP)",
            "Treinamento de motoristas (>5 anos pref.)",
            "Acordos de contraentrega com Top-20 clientes",
            "Dashboard tempo-real para despachante",
        ],
        "Redução_TVV_pts": [6.0, 4.5, 2.5, 3.0, 2.0],
        "Investimento_R$k": [180, 250, 45, 20, 80],
        "Prazo_meses":      [3, 5, 1, 2, 4],
        "Prob_Sucesso_pct":  [80, 75, 95, 90, 85],
    })

    custo_mensal_vazio = kpis["Custo Total Vazio (R$ M)"] * 1e6
    tvv_base           = kpis["Taxa Viagem Vazia (%)"] / 100

    # Monte Carlo — 10.000 simulações de ROI por ação
    np.random.seed(42)
    N = 10_000
    rois = []
    for _, row in cenarios.iterrows():
        reducao_sim = np.random.normal(row["Redução_TVV_pts"],
                                        row["Redução_TVV_pts"] * 0.20, N)
        economia_sim = (reducao_sim / 100) * custo_mensal_vazio * 12
        roi_sim = (economia_sim - row["Investimento_R$k"] * 1000) / \
                   (row["Investimento_R$k"] * 1000) * 100
        rois.append({
            "ROI_médio_%": roi_sim.mean(),
            "ROI_p10_%":   np.percentile(roi_sim, 10),
            "ROI_p90_%":   np.percentile(roi_sim, 90),
            "Economia_R$k_ano": economia_sim.mean() / 1000,
        })

    df_roi = pd.concat([cenarios, pd.DataFrame(rois)], axis=1)
    df_roi = df_roi.sort_values("ROI_médio_%", ascending=False)

    print("\n  🎯 Cenários de Melhoria com ROI (Monte Carlo, N=10.000):")
    print("  " + "─"*68)
    print(f"    {'Ação':<42} {'Invest.':>8} {'ROI Médio':>10} {'Economia/ano':>13}")
    print("  " + "─"*68)
    for _, r in df_roi.iterrows():
        print(f"    {r['Ação'][:40]:<42} "
              f"R${r['Investimento_R$k']:>5.0f}k "
              f"{r['ROI_médio_%']:>9.0f}% "
              f"R${r['Economia_R$k_ano']:>8.0f}k/ano")
    print("  " + "─"*68)

    total_reducao = cenarios["Redução_TVV_pts"].sum()
    tvv_nova      = max(kpis["Taxa Viagem Vazia (%)"] - total_reducao, 0)
    print(f"\n  ✅ Com todas as ações: TVV {kpis['Taxa Viagem Vazia (%)']:.1f}% "
          f"→ {tvv_nova:.1f}% (meta: ≤ 22%)")
    print(f"  💰 Economia projetada total: "
          f"R$ {df_roi['Economia_R$k_ano'].sum()/1000:.2f} M/ano\n")

    return df_roi


# ═══════════════════════════════════════════════════════════════════════════════
# FASE C — CONTROL (CEP)
# ═══════════════════════════════════════════════════════════════════════════════
def fase_control(df: pd.DataFrame):
    """
    Cria carta de controle X-bar para TVV semanal.
    Plano de controle com LSC, LC e LIC.
    """
    print("═"*65)
    print("  FASE C — CONTROL")
    print("═"*65)

    # Agrega TVV por semana
    semanal = df.groupby("semana").agg(
        TVV=("viagem_vazia","mean"),
        n=("viagem_vazia","count"),
    ).reset_index()
    semanal["TVV_pct"] = semanal["TVV"] * 100

    # Parâmetros da carta X-bar (3 sigma)
    media = semanal["TVV_pct"].mean()
    std   = semanal["TVV_pct"].std()
    lsc   = media + 3 * std
    lic   = max(media - 3 * std, 0)

    # Identifica pontos fora de controle
    semanal["fora_controle"] = (
        (semanal["TVV_pct"] > lsc) | (semanal["TVV_pct"] < lic)
    )

    n_fora = semanal["fora_controle"].sum()
    print(f"\n  📊 Carta de Controle X-bar — TVV Semanal:")
    print(f"     LC  (Linha Central) : {media:.2f}%")
    print(f"     LSC (Lim. Sup. Contr): {lsc:.2f}%")
    print(f"     LIC (Lim. Inf. Contr): {lic:.2f}%")
    print(f"     Pontos fora de controle: {n_fora} semanas")

    print("""
  📋 PLANO DE CONTROLE — TVV (Taxa de Viagem Vazia)
  ─────────────────────────────────────────────────────────────
  Métrica         : Taxa Viagem Vazia (%)
  Frequência      : Semanal (todo 2ª feira, 08h)
  Responsável     : Coordenador de Operações
  Ferramenta      : Dashboard Power BI + Alerta E-mail
  
  REAÇÃO se TVV > LSC ({:.1f}%):
    1. Reunião emergencial (até 4h após alerta)
    2. Ativar plataforma de carga de retorno
    3. Contactar comercial para redistribuição de cargas
    4. Registrar causa-raiz no sistema de qualidade
  
  REAÇÃO se TVV > meta + 3pp por 2 semanas consecutivas:
    1. Acionar revisão completa do plano de controle
    2. Retornar à fase Analyze (causa-raiz adicional)
  ─────────────────────────────────────────────────────────────
""".format(lsc))

    return semanal, media, lsc, lic


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES CONSOLIDADAS
# ═══════════════════════════════════════════════════════════════════════════════
def gerar_dashboard(df: pd.DataFrame, df_pareto: pd.DataFrame,
                     df_roi: pd.DataFrame, semanal: pd.DataFrame,
                     media: float, lsc: float, lic: float, kpis: dict):
    """Dashboard DMAIC completo em uma figura."""
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Lean Six Sigma — Redução de Custo de Transporte Vazio\n"
                 "Dashboard DMAIC", fontsize=17, fontweight="bold",
                 color="#1A237E", y=0.99)

    # --- 1: KPIs (tabela visual) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    kpi_items = [
        ("TVV Baseline",   f"{kpis['Taxa Viagem Vazia (%)']:.1f}%",   VERMELHO),
        ("Meta TVV",       "≤ 22.0%",                                   VERDE),
        ("Custo Vazio",    f"R$ {kpis['Custo Total Vazio (R$ M)']:.2f}M/mês", AMARELO),
        ("Nível Sigma",    f"{kpis['Nível Sigma']:.2f} σ",             AZUL),
        ("DPMO",           f"{kpis['DPMO']:,.0f}",                      CINZA),
    ]
    for i, (label, valor, cor) in enumerate(kpi_items):
        y_pos = 0.88 - i * 0.18
        ax1.add_patch(mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.07), 0.9, 0.15, boxstyle="round,pad=0.02",
            facecolor=cor, alpha=0.12, edgecolor=cor, linewidth=2,
            transform=ax1.transAxes
        ))
        ax1.text(0.10, y_pos + 0.01, label, transform=ax1.transAxes,
                 fontsize=10, color=CINZA)
        ax1.text(0.90, y_pos + 0.01, valor, transform=ax1.transAxes,
                 fontsize=11, fontweight="bold", color=cor, ha="right")
    ax1.set_title("KPIs — Baseline", fontweight="bold", color="#1A237E", pad=12)

    # --- 2: Pareto ---
    ax2 = fig.add_subplot(gs[0, 1])
    causas_curtas = [c[:32] for c in df_pareto["Causa"]]
    bars = ax2.bar(range(len(df_pareto)), df_pareto["Frequência_%"],
                   color=[AZUL,AZUL,AZUL,AMARELO,AMARELO,CINZA,CINZA],
                   edgecolor="white", alpha=0.85)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(df_pareto)), df_pareto["Acumulado_%"],
                  "ro-", linewidth=2, markersize=5, alpha=0.9)
    ax2_twin.axhline(80, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
    ax2_twin.set_ylabel("Acumulado (%)", color="red")
    ax2_twin.set_ylim(0, 115)
    ax2.set_xticks(range(len(df_pareto)))
    ax2.set_xticklabels(causas_curtas, rotation=35, ha="right", fontsize=7.5)
    ax2.set_ylabel("Frequência (%)")
    ax2.set_title("Pareto de Causas", fontweight="bold", color="#1A237E")

    # --- 3: ROI por ação ---
    ax3 = fig.add_subplot(gs[0, 2])
    acoes_curtas = [a[:30] for a in df_roi["Ação"]]
    cores_roi    = [VERDE if r > 200 else AMARELO if r > 100 else VERMELHO
                    for r in df_roi["ROI_médio_%"]]
    ax3.barh(acoes_curtas, df_roi["ROI_médio_%"], color=cores_roi,
             edgecolor="white", alpha=0.85, xerr=(
                 df_roi["ROI_médio_%"]-df_roi["ROI_p10_%"],
                 df_roi["ROI_p90_%"]-df_roi["ROI_médio_%"]
             ), error_kw={"ecolor":"gray","elinewidth":1.5})
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_xlabel("ROI (%)")
    ax3.set_title("ROI por Ação de Melhoria\n(Monte Carlo ±IC80%)",
                  fontweight="bold", color="#1A237E")

    # --- 4: Carta de controle X-bar ---
    ax4 = fig.add_subplot(gs[1, :2])
    x = semanal["semana"]
    y = semanal["TVV_pct"]
    ax4.plot(x, y, color=AZUL, linewidth=1.8, alpha=0.85, label="TVV Semanal")
    ax4.axhline(media, color="green",   linestyle="-",  linewidth=2,
                label=f"LC = {media:.1f}%")
    ax4.axhline(lsc,   color=VERMELHO,  linestyle="--", linewidth=1.8,
                label=f"LSC = {lsc:.1f}%")
    ax4.axhline(lic,   color=VERMELHO,  linestyle="--", linewidth=1.8,
                label=f"LIC = {lic:.1f}%")
    ax4.axhline(22,    color=VERDE,     linestyle=":",  linewidth=2,
                label="Meta = 22%")
    fora = semanal[semanal["fora_controle"]]
    ax4.scatter(fora["semana"], fora["TVV_pct"], color=VERMELHO,
                s=80, zorder=5, label="Fora de controle")
    ax4.fill_between(x, lic, lsc, alpha=0.06, color=AZUL)
    ax4.set_xlabel("Semana do Ano")
    ax4.set_ylabel("TVV (%)")
    ax4.set_title("Carta de Controle X-bar — Taxa de Viagem Vazia (%)",
                  fontweight="bold", color="#1A237E")
    ax4.legend(fontsize=8.5, ncol=3)

    # --- 5: Simulação pós-melhoria ---
    ax5 = fig.add_subplot(gs[1, 2])
    fases   = ["Baseline\n(Atual)", "Após 3 meses\n(parcial)", "Após 6 meses\n(meta)"]
    tvv_sim = [kpis["Taxa Viagem Vazia (%)"], 30.5, 21.8]
    cores5  = [VERMELHO, AMARELO, VERDE]
    bars5   = ax5.bar(fases, tvv_sim, color=cores5, edgecolor="white",
                      alpha=0.85, width=0.5)
    ax5.axhline(22, color="green", linestyle="--", linewidth=2, label="Meta 22%")
    for bar, val in zip(bars5, tvv_sim):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                 f"{val:.1f}%", ha="center", fontweight="bold",
                 fontsize=11, color="black")
    ax5.set_ylabel("TVV (%)")
    ax5.set_ylim(0, 45)
    ax5.set_title("Projeção de Melhoria TVV\n(Baseline → Meta)",
                  fontweight="bold", color="#1A237E")
    ax5.legend(fontsize=9)

    plt.savefig(f"{OUTPUT_DIR}/lean6sigma_dmaic_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    print(f"\n✅ Dashboard DMAIC salvo em: {OUTPUT_DIR}/lean6sigma_dmaic_dashboard.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PROJETO 3 — Lean Six Sigma: Redução de Transporte Vazio")
    print("=" * 65)

    fase_define()

    print("\n[Gerando dados operacionais...]")
    df = gerar_dados()
    print(f"  {len(df):,} registros | {df['veiculo'].nunique()} veículos "
          f"| {df['rota'].nunique()} rotas")

    kpis    = fase_measure(df)
    pareto  = fase_analyze(df)
    roi     = fase_improve(kpis)
    semanal, media, lsc, lic = fase_control(df)

    print("[Gerando dashboard visual...]")
    gerar_dashboard(df, pareto, roi, semanal, media, lsc, lic, kpis)

    # Exportar tabelas
    df.to_csv(f"{OUTPUT_DIR}/dados_operacionais.csv",
              sep=";", decimal=",", encoding="utf-8-sig", index=False)
    roi.to_csv(f"{OUTPUT_DIR}/plano_melhoria_roi.csv",
               sep=";", decimal=",", encoding="utf-8-sig", index=False)
    semanal.to_csv(f"{OUTPUT_DIR}/carta_controle_semanal.csv",
                   sep=";", decimal=",", encoding="utf-8-sig", index=False)

    print("\n" + "=" * 65)
    print("  ✅ PROJETO CONCLUÍDO COM SUCESSO!")
    print(f"  📁 Resultados em: ./{OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
