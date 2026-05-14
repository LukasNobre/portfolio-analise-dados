# 📊 Portfólio de Análise de Dados — Python

> Três projetos de nível profissional demonstrando competências em **Análise de Dados**, **Pesquisa Operacional** e **Melhoria de Processos**.

---

## 🗂️ Estrutura do Portfólio

```
portfolio/
├── requirements.txt
│
├── projeto1_powerbi/
│   └── analise_dados_powerbi.py          ← Pipeline ETL + Exportação Power BI
│
├── projeto2_dea_portuario/
│   └── dea_portuario.py                  ← DEA-CCR/BCC + Tendências Portuárias
│
└── projeto3_lean6sigma/
    └── lean6sigma_transporte_vazio.py    ← DMAIC + CEP + Monte Carlo
```

---

## 🚀 Como Executar

```bash
# 1. Clone e acesse o repositório
git clone https://github.com/seu-usuario/portfolio-analise-dados.git
cd portfolio-analise-dados

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute os projetos
python projeto1_powerbi/analise_dados_powerbi.py
python projeto2_dea_portuario/dea_portuario.py
python projeto3_lean6sigma/lean6sigma_transporte_vazio.py
```

---

## 📁 Projeto 1 — Análise de Dados com Exportação para Power BI

### Objetivo
Pipeline completo de análise de dados de vendas com limpeza, transformação, geração de KPIs e exportação de datasets estruturados (modelo estrela) para consumo no Power BI.

### Competências Demonstradas
| Área | Tecnologia/Técnica |
|------|-------------------|
| ETL | `pandas` — limpeza, tipagem, enriquecimento temporal |
| KPIs | Margem, ticket médio, EBITDA simplificado |
| Modelagem | Modelo estrela (fato + dimensões) |
| Visualização | `matplotlib` + `seaborn` — 6 gráficos analíticos |
| Exportação | CSV delimitado por ponto-e-vírgula para Power BI |

### Outputs Gerados
```
output_powerbi/
├── fato_vendas.csv          ← Tabela fato (5.000 registros)
├── dim_tempo.csv            ← Dimensão temporal
├── dim_geografia.csv        ← Dimensão regional
├── kpis_executivo.csv       ← KPIs prontos para cartões no PBI
├── resumo_mensal.csv        ← Agregações mensais
└── dashboard_vendas.png     ← Dashboard estático (6 painéis)
```

### Como Usar no Power BI
1. Abrir Power BI Desktop → **Obter Dados → Texto/CSV**
2. Importar os 5 arquivos CSV (delimitador: `;`, decimal: `,`)
3. No modelo de dados, criar relacionamentos entre `fato_vendas` e as dimensões
4. Construir os visuais usando os campos e medidas

---

## 📁 Projeto 2 — Análise de Eficiência Portuária com DEA

### Objetivo
Avaliar a eficiência operacional de 12 terminais portuários brasileiros usando **Data Envelopment Analysis (DEA)** nos modelos CCR (retornos constantes) e BCC (retornos variáveis), com análise de tendência temporal e geração de planos de melhoria.

### Referências Acadêmicas
- **Charnes, Cooper & Rhodes (1978)** — Measuring the efficiency of decision making units. *European Journal of Operational Research*.
- **Banker, Charnes & Cooper (1984)** — Some models for estimating technical and scale inefficiencies in DEA. *Management Science*.

### Competências Demonstradas
| Área | Tecnologia/Técnica |
|------|-------------------|
| Pesquisa Operacional | DEA-CCR e DEA-BCC via `scipy.optimize.linprog` |
| Séries Temporais | Análise de tendência + regressão linear |
| Benchmark | Identificação de DMUs de referência e lambdas |
| Gap Analysis | Folgas de input/output e metas de melhoria |
| Visualização | 5 painéis: ranking, CCR vs BCC, escala, tendências, gap |

### Variáveis do Modelo

**Inputs (recursos):**
- Trabalhadores (nº de funcionários)
- Equipamentos (guindastes, reach stackers)
- Área de pátio (hectares)
- Custo operacional (R$ milhões/ano)

**Outputs (resultados):**
- TEU movimentados (mil/ano)
- Receita bruta (R$ milhões/ano)
- Taxa de ocupação de berço (%)
- NPS dos armadores (0-100)

### Outputs Gerados
```
output_dea_portuario/
├── resultados_dea.csv              ← Escores CCR + plano de melhoria
├── tendencias_portuarias.csv       ← Tendências por terminal
└── dea_portuario_dashboard.png     ← Dashboard 5 painéis
```

---

## 📁 Projeto 3 — Lean Six Sigma: Redução de Transporte Vazio

### Objetivo
Projeto completo de **Lean Six Sigma** seguindo o ciclo **DMAIC** para reduzir a taxa de viagens vazias de uma transportadora de cargas, com economias projetadas de R$ 1,2 M/ano.

### Metodologia DMAIC Implementada

| Fase | Conteúdo | Técnica |
|------|----------|---------|
| **D**efine | Project Charter, SIPOC | Mapeamento de processo |
| **M**easure | Baseline, KPIs, DPMO | Nível Sigma, Estatística descritiva |
| **A**nalyze | Causa-raiz, correlações | Pareto 80/20, Pearson, regressão |
| **I**mprove | Cenários de melhoria | Simulação Monte Carlo (N=10.000) |
| **C**ontrol | Carta de controle X-bar | CEP com LSC/LC/LIC |

### KPIs do Projeto
| Indicador | Baseline | Meta | Após 6 meses |
|-----------|----------|------|--------------|
| Taxa Viagem Vazia | ~38% | ≤ 22% | 21,8% |
| Nível Sigma | ~2.5σ | ≥ 4.0σ | ~3.8σ |
| Custo Vazio Mensal | R$ 1,85M | R$ 1,10M | R$ 1,08M |

### Competências Demonstradas
| Área | Tecnologia/Técnica |
|------|-------------------|
| Lean Six Sigma | DMAIC completo, Project Charter, SIPOC |
| Estatística | CEP, DPMO, Nível Sigma, distribuição normal |
| Simulação | Monte Carlo com 10.000 iterações |
| Análise de Causa | Diagrama de Pareto, correlações |
| Controle | Cartas X-bar com LSC/LC/LIC |

### Outputs Gerados
```
output_lean6sigma/
├── dados_operacionais.csv          ← 1.620 registros de viagens
├── plano_melhoria_roi.csv          ← Cenários com ROI (Monte Carlo)
├── carta_controle_semanal.csv      ← Dados CEP semanais
└── lean6sigma_dmaic_dashboard.png  ← Dashboard DMAIC completo
```

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| `pandas` | ≥ 2.0 | Manipulação e análise de dados |
| `numpy` | ≥ 1.24 | Computação numérica |
| `scipy` | ≥ 1.11 | Otimização linear (DEA), estatística |
| `matplotlib` | ≥ 3.7 | Visualizações e dashboards |
| `seaborn` | ≥ 0.12 | Gráficos estatísticos |

---

## 👤 Autor

**[Seu Nome]**  
📧 seu.email@email.com  
🔗 [LinkedIn](https://linkedin.com/in/seu-perfil)  
🐙 [GitHub](https://github.com/seu-usuario)

---

## 📄 Licença

MIT License — Livre para uso, modificação e distribuição com atribuição.
