# gerar_dashboard.py
# Executado automaticamente pelo GitHub Actions todo mes
# Baixa dados novos da ANTAQ e atualiza o dashboard HTML

import os, sys, requests, zipfile, io
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import linprog
from datetime import datetime

# Download automatico dos dados ANTAQ
BASE_URL = "https://web3.antaq.gov.br/ea/txt"
ANO_ATUAL = datetime.now().year

os.makedirs("Dados Atracacao", exist_ok=True)
os.makedirs("Dados Carga", exist_ok=True)

for ano in range(2015, ANO_ATUAL + 1):
    for tipo, pasta in [("Atracacao", "Dados Atracacao"), ("Carga", "Dados Carga")]:
        arq = f"{pasta}/{ano}{tipo}.txt"
        if not os.path.exists(arq):
            try:
                print(f"Baixando {ano}{tipo}...")
                r = requests.get(f"{BASE_URL}/{ano}{tipo}.zip", timeout=300)
                if r.status_code == 200:
                    zipfile.ZipFile(io.BytesIO(r.content)).extractall(pasta)
                    print(f"OK: {ano}{tipo}")
                else:
                    print(f"Nao disponivel: {ano}{tipo}")
            except Exception as e:
                print(f"Erro {ano}{tipo}: {e}")
        else:
            print(f"Ja existe: {ano}{tipo}")

CAMINHO_ATRACACAO = "Dados Atracacao"
CAMINHO_CARGA = "Dados Carga"
OUTPUT_DIR = "."

"""
============================================================
ANÁLISE DE MOVIMENTAÇÃO PORTUÁRIA — SOJA (CDMercadoria 1201)
Complexo Portuário: Vila do Conde - Belém
Metodologia: Exploração + DEA (Data Envelopment Analysis)
============================================================

Fonte dos dados: ANTAQ — Agência Nacional de Transportes Aquaviários
Período analisado: 2015–2024 (10 anos)
Mercadoria: Soja (CDMercadoria 1201) — Granel Sólido — Embarcados

Autor: Lucas Nobre
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from scipy import stats
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# ─── Configurações visuais ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#F8FAFC",
    "axes.facecolor":   "#F8FAFC",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─── Caminhos das pastas de dados ─────────────────────────────────────────────
# Ajuste os caminhos abaixo conforme sua máquina:
CAMINHO_ATRACACAO = r"Dados Atracacao"
CAMINHO_CARGA     = r"Dados Carga"
OUTPUT_DIR        = r"."
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.options.display.max_columns = 50


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CARGA DOS DADOS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Tipos de colunas — Atracação ─────────────────────────────────────────────
dict_type_atracacao = {
    'IDAtracacao'                   : str,
    'CDTUP'                         : str,
    'IDBerco'                       : str,   # ← corrigido (era 'IDBercoBerço')
    'Berço'                         : str,
    'Porto Atracação'               : str,
    'Apelido Instalação Portuária'  : str,
    'Complexo Portuário'            : str,
    'Tipo da Autoridade Portuária'  : str,
    'Data Atracação'                : str,
    'Data Chegada'                  : str,
    'Data Desatracação'             : str,
    'Data Início Operação'          : str,
    'Data Término Operação'         : str,
    'Ano'                           : str,
    'Mes'                           : str,
    'Tipo de Operação'              : str,
    'Tipo de Navegação da Atracação': str,
    'Nacionalidade do Armador'      : str,
    'FlagMCOperacaoAtracacao'       : str,
    'Terminal'                      : str,
    'Município'                     : str,
    'UF'                            : str,
    'SGUF'                          : str,
    'Região Geográfica'             : str,
    'Nº da Capitania'               : str,
    'Nº do IMO'                     : str,
}

# ─── Tipos de colunas — Carga ─────────────────────────────────────────────────
dict_type_carga = {
    'IDCarga'                              : str,
    'IDAtracacao'                          : str,
    'Origem'                               : str,
    'Destino'                              : str,
    'CDMercadoria'                         : str,
    'Tipo Operação da Carga'               : str,
    'Carga Geral Acondicionamento'         : str,
    'ConteinerEstado'                      : str,
    'Tipo Navegação'                       : str,
    'FlagAutorizacao'                      : str,
    'FlagCabotagem'                        : str,
    'FlagCabotagemMovimentacao'            : str,
    'FlagConteinerTamanho'                 : str,
    'FlagLongoCurso'                       : str,
    'FlagMCOperacaoCarga'                  : str,
    'FlagOffshore'                         : str,
    'FlagTransporteViaInterioir'           : str,
    'Percurso Transporte em vias Interiores': str,
    'Percurso Transporte Interiores'       : str,
    'STNaturezaCarga'                      : str,
    'STSH2'                                : str,
    'STSH4'                                : str,
    'Natureza da Carga'                    : str,
    'Sentido'                              : str,
    'TEU'                                  : str,
    'QTCarga'                              : str,
    'VLPesoCargaBruta'                     : str,
}


def carregar_atracacao(caminho: str) -> pd.DataFrame:
    """Lê e concatena todos os arquivos .txt de Atracação."""
    dfs = []
    arquivos = [f for f in os.listdir(caminho) if f.endswith('.txt')]
    print(f"  Atracação: {len(arquivos)} arquivos encontrados")
    for arq in sorted(arquivos):
        path = os.path.join(caminho, arq)
        df = pd.read_csv(path, delimiter=';', dtype=dict_type_atracacao,
                         encoding='utf-8-sig')
        dfs.append(df)
        print(f"    ✔ {arq} — {len(df):,} linhas")
    df_concat = pd.concat(dfs, ignore_index=True)
    print(f"  Total Atracação: {len(df_concat):,} registros\n")
    return df_concat


def carregar_carga(caminho: str) -> pd.DataFrame:
    """Lê e concatena todos os arquivos .txt de Carga."""
    dfs = []
    arquivos = [f for f in os.listdir(caminho) if f.endswith('.txt')]
    print(f"  Carga: {len(arquivos)} arquivos encontrados")
    for arq in sorted(arquivos):
        path = os.path.join(caminho, arq)
        df = pd.read_csv(path, delimiter=';', dtype=dict_type_carga,
                         encoding='utf-8-sig')
        dfs.append(df)
        print(f"    ✔ {arq} — {len(df):,} linhas")
    df_concat = pd.concat(dfs, ignore_index=True)
    print(f"  Total Carga: {len(df_concat):,} registros\n")
    return df_concat


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — TIPAGEM, FILTROS E MERGE
# ═══════════════════════════════════════════════════════════════════════════════

def preparar_carga(df_carga: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos, aplica filtros de soja e agrupamento."""

    # Conversão numérica (vírgula → ponto)
    df_carga['VLPesoCargaBruta'] = (
        df_carga['VLPesoCargaBruta'].str.replace(',', '.').astype(float)
    )

    # ─── Filtros conforme análise original ────────────────────────────────────
    filtros = {
        'STSH4'           : 'Exclusivo',
        'Sentido'         : 'Embarcados',
        'Natureza da Carga': 'Granel Sólido',
        'CDMercadoria'    : '1201',          # Soja
    }

    df_filtrado = df_carga.loc[
        (df_carga['STSH4']            == filtros['STSH4']) &
        (df_carga['Sentido']          == filtros['Sentido']) &
        (df_carga['Natureza da Carga']== filtros['Natureza da Carga']) &
        (df_carga['CDMercadoria']     == filtros['CDMercadoria'])
    ].reset_index(drop=True)

    print(f"  Registros após filtro de soja: {len(df_filtrado):,}")

    # ─── Agrupamento para eliminar duplicatas por IDAtracacao ─────────────────
    colunas_grupo = [
        'IDAtracacao', 'CDMercadoria', 'Tipo Operação da Carga',
        'Tipo Navegação', 'Percurso Transporte Interiores',
        'STNaturezaCarga', 'STSH2', 'STSH4',
        'Natureza da Carga', 'Sentido', 'TEU', 'QTCarga',
    ]
    df_agrupado = (
        df_filtrado.groupby(colunas_grupo)['VLPesoCargaBruta']
        .sum()
        .reset_index()
    )
    print(f"  Registros após agrupamento:    {len(df_agrupado):,}\n")
    return df_agrupado


def enriquecer_com_atracacao(df_carga_ag: pd.DataFrame,
                              df_atracacao: pd.DataFrame) -> pd.DataFrame:
    """Merge entre carga agrupada e atracação."""
    df_merge = pd.merge(df_carga_ag, df_atracacao,
                        on='IDAtracacao', how='inner')
    print(f"  Registros após merge: {len(df_merge):,}")
    print(f"  Anos disponíveis:     {sorted(df_merge['Ano'].unique())}\n")
    return df_merge


def selecionar_porto(df_merge: pd.DataFrame,
                     complexo: str = 'Vila do Conde - Belém') -> pd.DataFrame:
    """Filtra pelo complexo portuário e seleciona colunas relevantes."""

    # Análise peso por complexo (mantida do original)
    print("  Peso total por Complexo Portuário:")
    resumo = (df_merge.groupby('Complexo Portuário')['VLPesoCargaBruta']
              .sum().sort_values(ascending=False))
    print(resumo.to_string())
    print()

    df_porto = df_merge.loc[
        df_merge['Complexo Portuário'] == complexo
    ].reset_index(drop=True)

    colunas_porto = [
        'IDAtracacao', 'CDMercadoria', 'Tipo Operação da Carga',
        'Tipo Navegação', 'Percurso Transporte Interiores', 'STSH4',
        'Natureza da Carga', 'Sentido', 'TEU', 'QTCarga',
        'VLPesoCargaBruta', 'Berço', 'Porto Atracação',
        'Complexo Portuário', 'Tipo da Autoridade Portuária',
        'Data Atracação', 'Data Chegada', 'Data Desatracação',
        'Data Início Operação', 'Data Término Operação',
        'Ano', 'Mes', 'Terminal', 'Município', 'UF', 'Região Geográfica',
    ]
    # Usa apenas colunas que existem no dataframe
    colunas_porto = [c for c in colunas_porto if c in df_porto.columns]
    df_porto = df_porto[colunas_porto]

    print(f"  Porto selecionado: {complexo}")
    print(f"  Registros:         {len(df_porto):,}")
    print(f"  Anos:              {sorted(df_porto['Ano'].unique())}")
    print(f"  Berços:            {df_porto['Berço'].unique().tolist()}")
    print(f"  Terminais:         {df_porto['Terminal'].unique().tolist()}\n")
    return df_porto


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CÁLCULOS OPERACIONAIS
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_indicadores(df_porto: pd.DataFrame) -> pd.DataFrame:
    """Calcula horas de operação e Prancha Operacional (t/h)."""
    fmt = '%d/%m/%Y %H:%M:%S'

    for col in ['Data Chegada', 'Data Atracação', 'Data Desatracação',
                'Data Início Operação', 'Data Término Operação']:
        df_porto[col] = pd.to_datetime(df_porto[col], format=fmt,
                                        errors='coerce')

    df_porto['Horas Atracação-Desatracação'] = (
        (df_porto['Data Desatracação'] - df_porto['Data Atracação'])
        .dt.total_seconds() / 3600
    )
    df_porto['Horas Início-Fim Operação'] = (
        (df_porto['Data Término Operação'] - df_porto['Data Início Operação'])
        .dt.total_seconds() / 3600
    )

    # Prancha Operacional: toneladas por hora de operação
    df_porto['Prancha Operacional'] = (
        df_porto['VLPesoCargaBruta'] /
        df_porto['Horas Início-Fim Operação'].replace(0, np.nan)
    )

    # Remove registros com horas negativas ou nulas
    df_porto = df_porto[
        (df_porto['Horas Início-Fim Operação'] > 0) &
        (df_porto['Prancha Operacional'] > 0)
    ].reset_index(drop=True)

    print(f"  Registros válidos com Prancha Operacional: {len(df_porto):,}")
    print(f"  Prancha média geral: {df_porto['Prancha Operacional'].mean():.1f} t/h\n")
    return df_porto


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — VISUALIZAÇÕES (mantidas do original + melhoradas)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_peso_por_berco_ano(df_porto: pd.DataFrame):
    """Gráfico original: peso por berço, subplots por ano."""
    df_grouped = (df_porto.groupby(['Ano', 'Berço'])['VLPesoCargaBruta']
                  .sum().reset_index())
    categorias_anos = sorted(df_grouped['Ano'].unique())
    num_anos = len(categorias_anos)

    fig, axs = plt.subplots(1, num_anos, figsize=(max(15, num_anos * 2.5), 5),
                             sharey=True, facecolor='#F8FAFC')
    fig.suptitle('Peso Embarcado de Soja por Berço — Vila do Conde/Belém',
                 fontweight='bold', fontsize=13)

    for i, ano in enumerate(categorias_anos):
        df_ano = df_grouped[df_grouped['Ano'] == ano]
        ax = axs[i] if num_anos > 1 else axs
        ax.bar(df_ano['Berço'], df_ano['VLPesoCargaBruta'] / 1e6,
               color='#1565C0', alpha=0.85, edgecolor='white')
        ax.set_title(f'Ano {ano}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Berço', fontsize=8)
        ax.set_ylabel('Peso (Mt)' if i == 0 else '', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_peso_berco_ano.png'),
                dpi=150, bbox_inches='tight')
    print("  ✔ Gráfico salvo: 01_peso_berco_ano.png")
    plt.show()


def plot_prancha_bercos(df_porto: pd.DataFrame):
    """Gráfico original: média da Prancha Operacional por berço ao longo dos anos."""
    df_grouped = (df_porto.groupby(['Ano', 'Berço'])['Prancha Operacional']
                  .mean().reset_index())

    bercos_escolhidos = df_porto['Berço'].unique().tolist()
    cores = ['#1565C0', '#2E7D32', '#F9A825', '#C62828', '#7B1FA2',
             '#00838F', '#4E342E', '#546E7A']

    plt.figure(figsize=(12, 6), facecolor='#F8FAFC')
    for berco, cor in zip(bercos_escolhidos, cores):
        df_berco = df_grouped[df_grouped['Berço'] == berco]
        if len(df_berco) > 0:
            plt.plot(df_berco['Ano'], df_berco['Prancha Operacional'],
                     marker='o', label=berco, color=cor,
                     linewidth=2, markersize=6)

    plt.xlabel('Ano', fontsize=11)
    plt.ylabel('Média da Prancha Operacional (t/h)', fontsize=11)
    plt.title('Média das Pranchas Operacionais por Berço — Vila do Conde/Belém',
              fontweight='bold', fontsize=13)
    plt.legend(title='Berço', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_prancha_bercos_anos.png'),
                dpi=150, bbox_inches='tight')
    print("  ✔ Gráfico salvo: 02_prancha_bercos_anos.png")
    plt.show()


def plot_tendencia_peso_anual(df_porto: pd.DataFrame):
    """Análise de tendência: volume total embarcado por ano + regressão."""
    anual = (df_porto.groupby('Ano')['VLPesoCargaBruta']
             .sum().reset_index())
    anual['Ano_num'] = anual['Ano'].astype(int)
    anual['Peso_Mt'] = anual['VLPesoCargaBruta'] / 1e6

    slope, intercept, r, p, se = stats.linregress(
        anual['Ano_num'], anual['Peso_Mt'])
    anual['Tendencia'] = slope * anual['Ano_num'] + intercept

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#F8FAFC')
    ax.bar(anual['Ano'], anual['Peso_Mt'], color='#1565C0',
           alpha=0.75, edgecolor='white', label='Volume anual')
    ax.plot(anual['Ano'], anual['Tendencia'], color='red',
            linewidth=2.5, linestyle='--', label=f'Tendência (R²={r**2:.3f})')

    ax.set_xlabel('Ano', fontsize=11)
    ax.set_ylabel('Volume Embarcado (Mton)', fontsize=11)
    ax.set_title('Tendência de Movimentação — Soja no Complexo Vila do Conde/Belém',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Anotação da tendência
    direcao = "crescente" if slope > 0 else "decrescente"
    ax.annotate(
        f'Tendência {direcao}: {slope:+.2f} Mt/ano\np-valor: {p:.4f}',
        xy=(0.02, 0.90), xycoords='axes fraction',
        fontsize=10, color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_tendencia_volume_anual.png'),
                dpi=150, bbox_inches='tight')
    print(f"  ✔ Gráfico salvo: 03_tendencia_volume_anual.png")
    print(f"     Tendência: {slope:+.2f} Mt/ano | R²={r**2:.3f} | p={p:.4f}\n")
    plt.show()


def plot_prancha_boxplot(df_porto: pd.DataFrame):
    """Distribuição da Prancha Operacional por berço."""
    bercos = df_porto['Berço'].unique()
    data   = [df_porto[df_porto['Berço'] == b]['Prancha Operacional'].dropna().values
              for b in bercos]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#F8FAFC')
    bp = ax.boxplot(data, labels=bercos, patch_artist=True, notch=False)
    cores = ['#1565C0', '#2E7D32', '#F9A825', '#C62828']
    for patch, cor in zip(bp['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)

    ax.set_xlabel('Berço', fontsize=11)
    ax.set_ylabel('Prancha Operacional (t/h)', fontsize=11)
    ax.set_title('Distribuição da Prancha Operacional por Berço',
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_prancha_boxplot.png'),
                dpi=150, bbox_inches='tight')
    print("  ✔ Gráfico salvo: 04_prancha_boxplot.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — DEA (Data Envelopment Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def preparar_dea(df_porto: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os dados por Berço e Ano para aplicar DEA.
    INPUT:  Horas médias de operação (recurso utilizado)
    OUTPUT: Prancha Operacional média (resultado gerado)
            Volume total embarcado (resultado gerado)
    """
    df_dea = df_porto.groupby(['Ano', 'Berço']).agg(
        Horas_Op_Media    = ('Horas Início-Fim Operação', 'mean'),
        Horas_Atrac_Media = ('Horas Atracação-Desatracação', 'mean'),
        Prancha_Media     = ('Prancha Operacional', 'mean'),
        Volume_Total_Mt   = ('VLPesoCargaBruta', lambda x: x.sum() / 1e6),
        N_Atracacoes      = ('IDAtracacao', 'count'),
    ).reset_index()

    # Remove valores inválidos
    df_dea = df_dea[
        (df_dea['Horas_Op_Media'] > 0) &
        (df_dea['Prancha_Media'] > 0) &
        (df_dea['Volume_Total_Mt'] > 0)
    ].reset_index(drop=True)

    return df_dea


def dea_ccr(inputs: np.ndarray, outputs: np.ndarray) -> dict:
    """
    Modelo DEA-CCR orientado a inputs (Charnes, Cooper & Rhodes, 1978).
    Minimiza uso de recursos para atingir mesmo nível de output.
    """
    n_dmu, n_inp = inputs.shape
    _, n_out     = outputs.shape
    escores      = np.zeros(n_dmu)
    lambdas_all  = np.zeros((n_dmu, n_dmu))

    for i in range(n_dmu):
        c = np.zeros(1 + n_dmu)
        c[0] = 1.0

        A_ub = np.zeros((n_inp + n_out, 1 + n_dmu))
        b_ub = np.zeros(n_inp + n_out)

        for k in range(n_inp):
            A_ub[k, 0]  = -inputs[i, k]
            A_ub[k, 1:] = inputs[:, k]
        for r in range(n_out):
            A_ub[n_inp + r, 1:]  = -outputs[:, r]
            b_ub[n_inp + r]      = -outputs[i, r]

        bounds = [(0, None)] * (1 + n_dmu)
        bounds[0] = (0, 1.0)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method='highs')

        if res.success:
            escores[i]     = min(res.x[0], 1.0)
            lambdas_all[i] = res.x[1:]
        else:
            escores[i] = np.nan

    return {'escores': escores, 'lambdas': lambdas_all}


def rodar_dea(df_porto: pd.DataFrame) -> pd.DataFrame:
    """Prepara dados, aplica DEA e retorna ranking de eficiência."""
    print("  Preparando dados para DEA...")
    df_dea = preparar_dea(df_porto)
    print(f"  DMUs (Berço-Ano): {len(df_dea)}")

    # Normalização min-max para estabilidade numérica
    col_inp = ['Horas_Op_Media', 'Horas_Atrac_Media']
    col_out = ['Prancha_Media', 'Volume_Total_Mt']

    X = df_dea[col_inp].values.astype(float)
    Y = df_dea[col_out].values.astype(float)

    # Normaliza
    X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
    Y_norm = (Y - Y.min(0)) / (Y.max(0) - Y.min(0) + 1e-9)
    X_norm = np.clip(X_norm, 1e-6, None)
    Y_norm = np.clip(Y_norm, 1e-6, None)

    res = dea_ccr(X_norm, Y_norm)

    df_dea['Escore_DEA'] = np.round(res['escores'] * 100, 2)
    df_dea['Status'] = pd.cut(
        df_dea['Escore_DEA'],
        bins=[0, 70, 85, 99.9, 100.01],
        labels=['❌ Crítico', '⚠️ Ineficiente', '🔶 Levemente Inef.', '✅ Eficiente']
    )
    df_dea = df_dea.sort_values(['Berço', 'Ano'])
    return df_dea


def plot_dea_resultados(df_dea: pd.DataFrame):
    """Visualização dos escores DEA por berço e evolução temporal."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#F8FAFC')
    fig.suptitle('Análise DEA — Eficiência Operacional dos Berços\n(Vila do Conde - Belém)',
                 fontweight='bold', fontsize=14)

    # Gráfico 1: Escore médio por berço
    ax1 = axes[0]
    media_berco = df_dea.groupby('Berço')['Escore_DEA'].mean().sort_values()
    cores = ['#C62828' if v < 85 else '#F9A825' if v < 100 else '#2E7D32'
             for v in media_berco]
    bars = ax1.barh(media_berco.index, media_berco.values,
                    color=cores, edgecolor='white', alpha=0.85)
    ax1.axvline(100, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlim(0, 110)
    ax1.set_xlabel('Escore DEA-CCR Médio (%)')
    ax1.set_title('Eficiência Média por Berço', fontweight='bold')
    for bar, val in zip(bars, media_berco):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    # Gráfico 2: Evolução do escore ao longo dos anos
    ax2 = axes[1]
    cores_berco = ['#1565C0', '#2E7D32', '#F9A825', '#C62828', '#7B1FA2']
    for berco, cor in zip(df_dea['Berço'].unique(), cores_berco):
        df_b = df_dea[df_dea['Berço'] == berco].sort_values('Ano')
        ax2.plot(df_b['Ano'], df_b['Escore_DEA'],
                 marker='o', label=berco, color=cor, linewidth=2)
    ax2.axhline(100, color='red', linestyle='--', linewidth=1.5,
                label='Fronteira eficiente')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Escore DEA-CCR (%)')
    ax2.set_title('Evolução da Eficiência por Berço', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_dea_eficiencia.png'),
                dpi=150, bbox_inches='tight')
    print("  ✔ Gráfico salvo: 05_dea_eficiencia.png")
    plt.show()


def gerar_plano_melhoria(df_dea: pd.DataFrame) -> pd.DataFrame:
    """
    Para berços ineficientes, calcula as metas de melhoria
    e exibe recomendações operacionais.
    """
    media_eficiente = df_dea[df_dea['Escore_DEA'] >= 99.9]['Prancha_Media'].mean()
    df_dea = df_dea.copy()
    df_dea['Meta_Prancha_t_h'] = (media_eficiente if not np.isnan(media_eficiente)
                                   else df_dea['Prancha_Media'].max())
    df_dea['Gap_Prancha_%'] = np.where(
        df_dea['Escore_DEA'] < 99.9,
        (df_dea['Meta_Prancha_t_h'] - df_dea['Prancha_Media']) /
        df_dea['Prancha_Media'] * 100, 0
    ).round(1)

    print("\n  📊 RANKING DEA — Eficiência por Berço e Ano:")
    print("  " + "─"*70)
    print(f"  {'Berço':<15} {'Ano':<6} {'Escore':>8} {'Prancha(t/h)':>13} "
          f"{'Gap Melhoria':>14}  {'Status'}")
    print("  " + "─"*70)
    for _, r in df_dea.sort_values(['Escore_DEA']).iterrows():
        gap = f"+{r['Gap_Prancha_%']:.1f}%" if r['Gap_Prancha_%'] > 0 else "—"
        print(f"  {r['Berço']:<15} {r['Ano']:<6} {r['Escore_DEA']:>7.1f}% "
              f"{r['Prancha_Media']:>12.1f}  {gap:>14}  {r['Status']}")
    print("  " + "─"*70)

    # Recomendações
    ineficientes = df_dea[df_dea['Escore_DEA'] < 85]
    if len(ineficientes) > 0:
        print("\n  🎯 RECOMENDAÇÕES OPERACIONAIS:")
        for berco in ineficientes['Berço'].unique():
            dados = ineficientes[ineficientes['Berço'] == berco]
            gap_medio = dados['Gap_Prancha_%'].mean()
            print(f"\n  Berço {berco} (Gap médio: +{gap_medio:.1f}% na prancha):")
            print(f"    ► Revisar sequência de carregamento (ganho estimado: 10-15%)")
            print(f"    ► Avaliar disponibilidade e manutenção de equipamentos")
            print(f"    ► Reduzir tempo de espera entre atracação e início de operação")
            print(f"    ► Benchmark com berço de maior escore como referência")

    return df_dea


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — EXPORTAÇÃO PARA POWER BI
# ═══════════════════════════════════════════════════════════════════════════════

def exportar_powerbi(df_porto: pd.DataFrame, df_dea: pd.DataFrame):
    """Exporta as bases tratadas em CSV para consumo no Power BI."""
    print("\n  📤 Exportando para Power BI...")

    # Base operacional completa
    df_porto.to_csv(
        os.path.join(OUTPUT_DIR, 'base_operacional_porto.csv'),
        sep=';', decimal=',', encoding='utf-8-sig', index=False
    )

    # Resumo anual por berço
    resumo = df_porto.groupby(['Ano', 'Berço', 'Terminal']).agg(
        Volume_Mt         = ('VLPesoCargaBruta', lambda x: x.sum()/1e6),
        N_Atracacoes      = ('IDAtracacao', 'count'),
        Prancha_Media     = ('Prancha Operacional', 'mean'),
        Horas_Op_Media    = ('Horas Início-Fim Operação', 'mean'),
    ).reset_index()
    resumo.to_csv(
        os.path.join(OUTPUT_DIR, 'resumo_anual_berco.csv'),
        sep=';', decimal=',', encoding='utf-8-sig', index=False
    )

    # Resultados DEA
    df_dea.to_csv(
        os.path.join(OUTPUT_DIR, 'resultados_dea_bercos.csv'),
        sep=';', decimal=',', encoding='utf-8-sig', index=False
    )

    arquivos = ['base_operacional_porto.csv', 'resumo_anual_berco.csv',
                'resultados_dea_bercos.csv']
    for arq in arquivos:
        p = os.path.join(OUTPUT_DIR, arq)
        kb = os.path.getsize(p) / 1024
        print(f"    ✔ {arq:<45} {kb:>7.1f} KB")
    print(f"\n  💡 Arquivos prontos para importar no Power BI em:\n  {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  ANÁLISE PORTUÁRIA — SOJA | VILA DO CONDE - BELÉM")
    print("  Metodologia: Exploração + DEA | 10 anos de dados")
    print("=" * 65)

    # ── 1. Carga dos dados ────────────────────────────────────────────────────
    print("\n[1/6] Carregando bases de dados...")
    df_atracacao = carregar_atracacao(CAMINHO_ATRACACAO)
    df_carga     = carregar_carga(CAMINHO_CARGA)

    # ── 2. Filtros e merge ────────────────────────────────────────────────────
    print("[2/6] Aplicando filtros e merge...")
    df_carga_ag = preparar_carga(df_carga)
    df_merge    = enriquecer_com_atracacao(df_carga_ag, df_atracacao)

    # ── 3. Seleção do porto ───────────────────────────────────────────────────
    print("[3/6] Selecionando complexo portuário...")
    df_porto = selecionar_porto(df_merge, complexo='Vila do Conde - Belém')

    # ── 4. Indicadores operacionais ───────────────────────────────────────────
    print("[4/6] Calculando indicadores operacionais...")
    df_porto = calcular_indicadores(df_porto)

    # ── 5. Visualizações ─────────────────────────────────────────────────────
    print("[5/6] Gerando visualizações...\n")
    plot_peso_por_berco_ano(df_porto)
    plot_prancha_bercos(df_porto)
    plot_tendencia_peso_anual(df_porto)
    plot_prancha_boxplot(df_porto)

    # ── 6. DEA ────────────────────────────────────────────────────────────────
    print("[6/6] Executando análise DEA...")
    df_dea = rodar_dea(df_porto)
    plot_dea_resultados(df_dea)
    df_dea = gerar_plano_melhoria(df_dea)

    # ── Exportação ────────────────────────────────────────────────────────────
    exportar_powerbi(df_porto, df_dea)

    print("\n" + "=" * 65)
    print("  ✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"  📁 Todos os arquivos em: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
