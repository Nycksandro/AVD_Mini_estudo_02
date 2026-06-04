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
FIXED_B_EST    = 20     # Usaremos 20 como base de divisão inicial para achar 'm'
T_CRITICO_T    = 1.96   # Para OBM com muitos lotes sobrepostos, usa-se a aproximação normal (z)

MSER_BLOCO_INI = 500_000
MSER_BLOCO_EXT = 200_000

IC_FIRST_CHECK = 5_000
IC_RECALC      = 1_000

DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

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

# ─── Regra de Parada com OBM (Overlapping Batch Means) ────────────────────────
def estimar_com_obm_stop(sim, d, pct_overlap):
    """
    Aplica o critério de parada OBM.
    pct_overlap: 1.0 (100% de overlap, shift=1), 0.50 (shift=m/2), 0.25 (shift=m*0.75)
    """
    n_pos = IC_FIRST_CHECK
    
    while True:
        total_geral_necessario = d + n_pos
        if total_geral_necessario > len(sim.samples):
            sim.gerar(total_geral_necessario - len(sim.samples) + IC_RECALC)
            
        dados_estacionarios = sim.samples[d:d + n_pos]
        Y = len(dados_estacionarios)
        
        # Define o tamanho base do lote m
        m = Y // FIXED_B_EST
        if m < 5: 
            n_pos += IC_RECALC
            continue
            
        # Define o deslocamento (shift) com base no nível de superposição escolhido
        if pct_overlap == 1.0:
            c = 1
        elif pct_overlap == 0.50:
            c = max(1, m // 2)
        else:  # 25% overlap -> avança 75% do tamanho do lote
            c = max(1, int(m * 0.75))
            
        # Montagem dos lotes sobrepostos usando somas móveis de tamanho m com passo c
        medias_lotes = []
        inicio = 0
        while inicio + m <= Y:
            lote_soma = sum(dados_estacionarios[inicio:inicio + m])
            medias_lotes.append(lote_soma / m)
            inicio += c
            
        B = len(medias_lotes)
        if B < 2:
            n_pos += IC_RECALC
            continue
            
        media_global = sum(dados_estacionarios) / Y
        
        # Variância base do estimador OBM ajustada pelo número de lotes e deslocamento
        soma_quadrados = sum((x - media_global) ** 2 for x in medias_lotes)
        
        # Variância da média amostral corrigida matematicamente para OBM
        var_media_global = (m * soma_quadrados) / (B * (Y - m))
        erro_padrao = math.sqrt(var_media_global)
        
        H = T_CRITICO_T * erro_padrao
        precisao_relativa = H / media_global if media_global > 0 else float("inf")
        
        if precisao_relativa <= GAMMA:
            return {
                "mean": media_global,
                "ic_lo": media_global - H,
                "ic_hi": media_global + H,
                "n_final": total_geral_necessario,
                "d": d,
                "m": m,
                "B": B
            }
            
        n_pos += IC_RECALC

# ─── Execução dos Cenários e Configurações de Overlap ────────────────────────
cenarios = [
    {"id": "I",   "lambda": 7.0,  "mu": 10.0},
    {"id": "II",  "lambda": 8.0,  "mu": 10.0},
    {"id": "III", "lambda": 9.0,  "mu": 10.0},
    {"id": "IV",  "lambda": 9.5,  "mu": 10.0}
]

overlaps = [
    {"rotulo": "100%", "valor": 1.0},
    {"rotulo": "50%",  "valor": 0.50},
    {"rotulo": "25%",  "valor": 0.25}
]

# Dicionário para encapsular os resultados finais
mapeamento_resultados = {ov["rotulo"]: [] for ov in overlaps}

print("=" * 80)
print(" Simulação M/M/1 com Remoção MSER-5Y e Parada por Overlapping Batch Means (OBM)")
print("=" * 80)

for ceno in cenarios:
    lam = ceno["lambda"]
    mu = ceno["mu"]
    rho = lam / mu
    wq_teorico = (rho / mu) / (1 - rho)
    
    print(f"\n[Cenário {ceno['id']}] (λ={lam}, μ={mu}, ρ={rho:.2f})")
    
    # Criamos o mesmo histórico de simulação base para testar de forma justa os overlaps
    sim_base = SimuladorMM1(lam, mu)
    sim_base.gerar(MSER_BLOCO_INI)
    
    d_est = mser5_warmup(sim_base)
    print(f"  -> Transiente Removido (MSER-5Y): d* = {d_est:,}")
    
    for ov in overlaps:
        print(f"     -> Calculando OBM com Overlap de {ov['rotulo']}...")
        res = estimar_com_obm_stop(sim_base, d_est, ov["valor"])
        res.update({"teorico": wq_teorico, "id": ceno["id"], "rho": rho})
        mapeamento_resultados[ov["rotulo"]].append(res)

# ─── Plotagem dos Gráficos Comparativos ────────────────────────────────────────
print("\n[Renderizando Gráficos]")

colors = ["#185FA5", "#0F6E56", "#D85A30", "#A32A2A"]

for rotulo, lista_res in mapeamento_resultados.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, res in enumerate(lista_res):
        ax = axes[i]
        
        # Barra do tempo médio estimado
        ax.bar([0], [res["mean"]], width=0.4, color=colors[i], alpha=0.85, edgecolor=colors[i], label="Estimado (OBM)")
        err_lo = res["mean"] - res["ic_lo"]
        err_hi = res["ic_hi"] - res["mean"]
        ax.errorbar([0], [res["mean"]], yerr=[[err_lo], [err_hi]], fmt="none", color="#2C2C2A", capsize=10, capthick=2, elinewidth=2)
        
        # Linha teórica esperada
        ax.axhline(res["teorico"], color="#E65100", linewidth=2, linestyle="--", label=f"Teórico ({res['teorico']:.4f} s)")
        
        # Título e legendas informativas
        ax.set_title(f"Cenário {res['id']} (ρ = {res['rho']:.2f})\nd*={res['d']:,} | m={res['m']:,} | Lotes Sobrepostos(B)={res['B']:,}\nN Final={res['n_final']:,}", fontsize=9, fontweight='bold')
        ax.set_ylabel("Wq (segundos)", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.6, zorder=0)
        
        txt_box = f"Estimado: {res['mean']:.4f} s\nIC: [{res['ic_lo']:.4f}, {res['ic_hi']:.4f}]"
        ax.text(0, res["ic_hi"] + (res["ic_hi"] * 0.05), txt_box, ha="center", va="bottom", fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
        
        ax.set_ylim(res["ic_lo"] * 0.7, res["ic_hi"] * 1.35)
        ax.legend(loc="lower right", fontsize=9)

    nome_arquivo = os.path.join(DIR, f"grafico_obm_overlap_{rotulo.replace('%','')}.png")
    plt.suptitle(f"Comparativo M/M/1 - Overlapping Batch Means (OBM) com {rotulo} de Overlap", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()
    print(f"  [Gráfico] Salvo com sucesso em: {nome_arquivo}")

print("\nProcesso OBM concluído!") 