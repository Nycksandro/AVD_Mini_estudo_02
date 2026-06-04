import csv
import math
import random
import time
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configurações Gerais ─────────────────────────────────────────────────────
GAMMA          = 0.05   # Precisão relativa de 5%
MSER_K         = 5      # Agrupamento para a curva MSER
B_LOTES        = 20     # Número fixo de lotes para o SBM
T_CRITICO_T    = 2.093  # t_0.025 para 19 graus de liberdade (B=20)

MSER_BLOCO_INI = 500_000
MSER_BLOCO_EXT = 200_000

IC_FIRST_CHECK = 2_000
IC_RECALC      = 500

DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
PNG_SAIDA_S1   = os.path.join(DIR, "grafico_sbm_S1.png")
PNG_SAIDA_SM10 = os.path.join(DIR, "grafico_sbm_Sm10.png")

# ─── Simulador M/M/1 ──────────────────────────────────────────────────────────
class SimuladorMM1:
    def __init__(self, taxa_entrada, taxa_servico):
        self.taxa_entrada = taxa_entrada
        self.taxa_servico = taxa_servico
        self.t        = 0.0
        self.queue    = 0
        self.busy     = False
        self.arrivals = deque()
        self.next_arr = random.expovariate(taxa_entrada)
        self.next_dep = float("inf")
        self.samples  = []

    def gerar(self, n):
        alvo = len(self.samples) + n
        while len(self.samples) < alvo:
            if self.next_arr <= self.next_dep:
                self.t = self.next_arr
                if not self.busy:
                    self.busy     = True
                    self.next_dep = self.t + random.expovariate(self.taxa_servico)
                    self.samples.append(0.0)
                else:
                    self.queue += 1
                    self.arrivals.append(self.t)
                self.next_arr = self.t + random.expovariate(self.taxa_entrada)
            else:
                self.t = self.next_dep
                if self.queue > 0:
                    self.queue   -= 1
                    wq            = self.t - self.arrivals.popleft()
                    self.samples.append(wq)
                    self.next_dep = self.t + random.expovariate(self.taxa_servico)
                else:
                    self.busy     = False
                    self.next_dep = float("inf")

# ─── MSER-5Y Algoritmo de Suporte ─────────────────────────────────────────────
def _mser_sufixo(batches, d_ini, d_fim):
    M = len(batches)
    sum_x  = [0.0] * (M + 1)
    sum_x2 = [0.0] * (M + 1)
    for i in range(M - 1, -1, -1):
        sum_x[i]  = sum_x[i+1]  + batches[i]
        sum_x2[i] = sum_x2[i+1] + batches[i] ** 2
    result = []
    for d in range(d_ini, min(d_fim, M - 2)):
        cnt  = M - d
        mu   = sum_x[d] / cnt
        mser = sum_x2[d] / cnt - mu * mu
        result.append((mser, d))
    return result

def mser5_warmup(sim, k=MSER_K):
    while True:
        samples    = sim.samples
        n          = len(samples)
        m          = n // k
        metade     = m // 2
        cinco_pct  = max(1, m // 20)
        fim_janela = metade + cinco_pct

        if m < 20:
            sim.gerar(MSER_BLOCO_EXT)
            continue

        batches = [sum(samples[b*k:(b+1)*k]) / k for b in range(m)]
        mser1        = _mser_sufixo(batches, 0,      metade)
        mser2_janela = _mser_sufixo(batches, metade, fim_janela)

        best_mser = min(mser1, key=lambda x: x[0])[0]
        melhores  = [d for mser, d in mser1 if mser == best_mser]
        min_mser2 = min((mser for mser, d in mser2_janela), default=float("inf"))

        if len(melhores) > 1 or min_mser2 < best_mser:
            sim.gerar(MSER_BLOCO_EXT)
            continue

        best_d = melhores[0]
        return best_d * k

# ─── Regra de Parada com SBM (Spaced Batch Means) ─────────────────────────────
def estimar_com_sbm_stop(sim, d, tipo_s):
    """
    Aplica o critério SBM com B=20 lotes.
    tipo_s: 'fixo' para S=1, 'proporcional' para S=m/10
    """
    n_pos = IC_FIRST_CHECK
    
    while True:
        # Determinar dinamicamente o tamanho do lote 'm' e do espaçamento 'S'
        if tipo_s == 'fixo':
            S = 1
            m = n_pos // B_LOTES
        else:  # proporcional: S = m / 10 -> n_pos = 20*m + 19*(m/10) = 21.9 * m
            m = int(n_pos / (B_LOTES + (B_LOTES - 1) / 10.0))
            S = max(1, m // 10)
            
        total_estacionario_necessario = B_LOTES * m + (B_LOTES - 1) * S
        total_geral_necessario = d + total_estacionario_necessario
        
        # Se precisarmos de mais dados, expande a simulação
        if total_geral_necessario > len(sim.samples):
            sim.gerar(total_geral_necessario - len(sim.samples) + IC_RECALC)
            
        dados_estacionarios = sim.samples[d:d + total_estacionario_necessario]
        
        # Extração dos lotes pulando os espaçamentos S
        medias_lotes = []
        ponteiro = 0
        for b in range(B_LOTES):
            inicio_lote = ponteiro
            fim_lote = ponteiro + m
            lote = dados_estacionarios[inicio_lote:fim_lote]
            medias_lotes.append(sum(lote) / m)
            
            # Avança o ponteiro pulando o lote e o espaçamento S subsequente
            ponteiro += m + S
            
        # Cálculos estatísticos clássicos sobre as médias dos lotes do SBM
        media_global = sum(medias_lotes) / B_LOTES
        var_lotes = sum((x - media_global) ** 2 for x in medias_lotes) / (B_LOTES - 1)
        erro_padrao = math.sqrt(var_lotes / B_LOTES)
        
        H = T_CRITICO_T * erro_padrao
        precisao_relativa = H / media_global if media_global > 0 else float("inf")
        
        if precisao_relativa <= GAMMA and m > 0:
            return {
                "mean": media_global,
                "ic_lo": media_global - H,
                "ic_hi": media_global + H,
                "n_final": total_geral_necessario,
                "d": d,
                "m": m,
                "S": S
            }
            
        n_pos += IC_RECALC

# ─── Configuração de Cenários e Execução ──────────────────────────────────────
cenarios = [
    {"id": "I",   "lambda": 7.0,  "mu": 10.0},
    {"id": "II",  "lambda": 8.0,  "mu": 10.0},
    {"id": "III", "lambda": 9.0,  "mu": 10.0},
    {"id": "IV",  "lambda": 9.5,  "mu": 10.0}
]

resultados_S1 = []
resultados_Sm10 = []

print("=" * 75)
print(" Executando Simulação M/M/1 com Remoção MSER-5Y e Parada por SBM (B=20)")
print("=" * 75)

for ceno in cenarios:
    lam = ceno["lambda"]
    mu = ceno["mu"]
    rho = lam / mu
    wq_teorico = (rho / mu) / (1 - rho)
    
    print(f"\n[Cenário {ceno['id']}] (λ={lam}, μ={mu}, ρ={rho:.2f})")
    
    # Instancia simulador base para o cenário
    sim_base = SimuladorMM1(lam, mu)
    sim_base.gerar(MSER_BLOCO_INI)
    
    # Ambas as estratégias usam o mesmo tamanho de transiente d para o mesmo seed/corrida
    d_est = mser5_warmup(sim_base)
    print(f"  -> Transiente detectado (MSER-5Y): d* = {d_est:,}")
    
    # Teste 1: SBM com S = 1
    print("  -> Calculando SBM com S = 1...")
    res_s1 = estimar_com_sbm_stop(sim_base, d_est, tipo_s='fixo')
    res_s1.update({"teorico": wq_teorico, "id": ceno["id"], "rho": rho})
    resultados_S1.append(res_s1)
    
    # Teste 2: SBM com S = m/10
    print("  -> Calculando SBM com S = m/10...")
    res_sm10 = estimar_com_sbm_stop(sim_base, d_est, tipo_s='proporcional')
    res_sm10.update({"teorico": wq_teorico, "id": ceno["id"], "rho": rho})
    resultados_Sm10.append(res_sm10)

# ─── Função de Plotagem customizada ──────────────────────────────────────────
def plotar_cenarios(resultados, path_salvamento, titulo_grafico):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ["#185FA5", "#0F6E56", "#D85A30", "#A32A2A"]

    for i, res in enumerate(resultados):
        ax = axes[i]
        
        # Barra de dados Estimados
        ax.bar([0], [res["mean"]], width=0.4, color=colors[i], alpha=0.85, edgecolor=colors[i], label="Estimado (SBM)")
        err_lo = res["mean"] - res["ic_lo"]
        err_hi = res["ic_hi"] - res["mean"]
        ax.errorbar([0], [res["mean"]], yerr=[[err_lo], [err_hi]], fmt="none", color="#2C2C2A", capsize=10, capthick=2, elinewidth=2)
        
        # Linha Teórica de Controle
        ax.axhline(res["teorico"], color="#E65100", linewidth=2, linestyle="--", label=f"Teórico ({res['teorico']:.4f} s)")
        
        # Ajustes de Títulos e Metadados exibidos
        ax.set_title(f"Cenário {res['id']} (ρ = {res['rho']:.2f})\nd*={res['d']:,} | m={res['m']:,} | S={res['S']:,} | N={res['n_final']:,}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Wq (segundos)", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.6, zorder=0)
        
        txt_box = f"Estimado: {res['mean']:.4f} s\nIC: [{res['ic_lo']:.4f}, {res['ic_hi']:.4f}]"
        ax.text(0, res["ic_hi"] + (res["ic_hi"] * 0.05), txt_box, ha="center", va="bottom", fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
        
        ax.set_ylim(res["ic_lo"] * 0.7, res["ic_hi"] * 1.3)
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle(titulo_grafico, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(path_salvamento, dpi=150)
    plt.close()
    print(f"  [Gráfico] Salvo em: {path_salvamento}")

# Gerando as saídas gráficas separadas por valor de S
print("\n[Renderizando Gráficos]")
plotar_cenarios(resultados_S1, PNG_SAIDA_S1, "Comparativo M/M/1 - Spaced Batch Means (SBM) com S = 1")
plotar_cenarios(resultados_Sm10, PNG_SAIDA_SM10, "Comparativo M/M/1 - Spaced Batch Means (SBM) com S = m/10")

print("\nProcesso concluído com sucesso!")