import csv
import math
import random
import time
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Configurações Gerais ─────────────────────────────────────────────────────
GAMMA          = 0.05  # Precisão relativa de 5%
MSER_K         = 5     # Agrupamento para a curva MSER
B_LOTES        = 20    # Número fixo de lotes para o NBM
T_CRITICO_T    = 2.093  # t_0.025 para 19 graus de liberdade (B=20)

MSER_BLOCO_INI = 500_000   # Reduzido ligeiramente para agilizar cenários leves
MSER_BLOCO_EXT = 200_000

IC_FIRST_CHECK = 2_000     # Amostras mínimas pós-transiente para começar a testar NBM
IC_RECALC      = 500

DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
PNG_SAIDA   = os.path.join(DIR, "grafico_nbm_cenarios.png")

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
    tentativa = 0
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
            tentativa += 1
            sim.gerar(MSER_BLOCO_EXT)
            continue

        best_d = melhores[0]
        return best_d * k

# ─── Regra de Parada com NBM (Nonoverlapping Batch Means) ────────────────────
def estimar_com_nbm_stop(sim, d):
    """
    Consome dados pós-transiente d. A cada incremento, recalcula os lotes do NBM
    usando B fixo (20) e testa a precisão contra GAMMA baseado na variância do lote.
    """
    n_pos = IC_FIRST_CHECK
    
    while True:
        # Garante que temos amostras suficientes na simulação
        total_necessario = d + n_pos
        if total_necessario > len(sim.samples):
            sim.gerar(total_necessario - len(sim.samples) + IC_RECALC)

        dados_estacionarios = sim.samples[d:d + n_pos]
        
        # Tamanho de cada um dos B lotes
        m = n_pos // B_LOTES
        
        # Calcula a média de cada lote do NBM
        medias_lotes = []
        for b in range(B_LOTES):
            inicio_lote = b * m
            fim_lote = (b + 1) * m
            medias_lotes.append(sum(dados_estacionarios[inicio_lote:fim_lote]) / m)
        
        # Média global e Variância das médias dos lotes
        media_global = sum(medias_lotes) / B_LOTES
        var_lotes = sum((x - media_global) ** 2 for x in medias_lotes) / (B_LOTES - 1)
        
        # Erro padrão da média global via NBM
        erro_padrao = math.sqrt(var_lotes / B_LOTES)
        
        # Semáforo de Precisão (Intervalo de Confiança t-Student)
        H = T_CRITICO_T * erro_padrao
        precisao_relativa = H / media_global if media_global > 0 else float("inf")
        
        if precisao_relativa <= GAMMA:
            return {
                "mean": media_global,
                "ic_lo": media_global - H,
                "ic_hi": media_global + H,
                "n_final": d + n_pos,
                "d": d
            }
        
        # Caso contrário, avança a janela de teste
        n_pos += IC_RECALC

# ─── Execução dos Cenários ────────────────────────────────────────────────────
cenarios = [
    {"id": "I",   "lambda": 7.0,  "mu": 10.0},
    {"id": "II",  "lambda": 8.0,  "mu": 10.0},
    {"id": "III", "lambda": 9.0,  "mu": 10.0},
    {"id": "IV",  "lambda": 9.5,  "mu": 10.0}
]

resultados = []

print("=" * 70)
print(f" Executando Simulação M/M/1 com Remoção MSER-5Y e Parada NBM (B={B_LOTES})")
print("=" * 70)

for ceno in cenarios:
    lam = ceno["lambda"]
    mu = ceno["mu"]
    rho = lam / mu
    wq_teorico = (rho / mu) / (1 - rho)
    
    print(f"\nRodando Cenário {ceno['id']} (λ={lam}, μ={mu}, ρ={rho:.2f})...")
    
    sim = SimuladorMM1(lam, mu)
    sim.gerar(MSER_BLOCO_INI)
    
    # 1. Eliminar transiente via MSER-5Y
    d_est = mser5_warmup(sim)
    
    # 2. Processar parada e estatísticas via NBM
    res = estimar_com_nbm_stop(sim, d_est)
    res["teorico"] = wq_teorico
    res["id"] = ceno["id"]
    res["rho"] = rho
    
    cobertura = "✓" if res["ic_lo"] <= wq_teorico <= res["ic_hi"] else "✗"
    print(f"  -> Concluído: d*={res['d']:,} | N_final={res['n_final']:,}")
    print(f"  -> Estimado: {res['mean']:.5f} s | Teórico: {wq_teorico:.5f} s [{cobertura}]")
    
    resultados.append(res)

# ─── Plotagem dos Gráficos Comparativos ────────────────────────────────────────
print("\n[Plotando] Gerando gráfico comparativo...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ["#185FA5", "#0F6E56", "#D85A30", "#A32A2A"]

for i, res in enumerate(resultados):
    ax = axes[i]
    
    # Renderização da barra do estimado com o IC associado
    ax.bar([0], [res["mean"]], width=0.4, color=colors[i], alpha=0.85, edgecolor=colors[i], label="Estimado (NBM)")
    err_lo = res["mean"] - res["ic_lo"]
    err_hi = res["ic_hi"] - res["mean"]
    ax.errorbar([0], [res["mean"]], yerr=[[err_lo], [err_hi]], fmt="none", color="#2C2C2A", capsize=10, capthick=2, elinewidth=2)
    
    # Linha Horizontal do Valor Teórico
    ax.axhline(res["teorico"], color="#E65100", linewidth=2, linestyle="--", label=f"Teórico ({res['teorico']:.4f} s)")
    
    # Ajustes finos de layout por quadrante
    ax.set_title(f"Cenário {res['id']} (ρ = {res['rho']:.2f})\nd* = {res['d']:,} | N = {res['n_final']:,}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Wq (segundos)", fontsize=10)
    ax.set_xticks([])
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.6, zorder=0)
    
    # Caixa de texto flutuante com dados resumidos
    txt_box = f"Estimado: {res['mean']:.4f} s\nIC: [{res['ic_lo']:.4f}, {res['ic_hi']:.4f}]"
    ax.text(0, res["ic_hi"] + (res["ic_hi"]*0.05), txt_box, ha="center", va="bottom", fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
    
    # Margem dinâmica para o topo do gráfico não cortar labels
    ax.set_ylim(res["ic_lo"] * 0.7, res["ic_hi"] * 1.25)
    ax.legend(loc="lower right", fontsize=9)

plt.suptitle("Comparativo de Cenários M/M/1 utilizando MSER-5Y e Parada por NBM (B=20)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(PNG_SAIDA, dpi=150)
plt.close()

print(f"\nProcesso Finalizado. Gráfico salvo com sucesso em: {PNG_SAIDA}")