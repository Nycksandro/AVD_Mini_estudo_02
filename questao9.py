import csv
import math
import random
import time
import os
from collections import deque
import scipy.stats as stats  # Utilizado para obter o t-student

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Parâmetros ─────────────────────────────
GAMMA          = 0.05   
B_NBM          = 20     
BETA_CRITICO   = 1.44   
CONFIDANCA     = 0.95

MSER_K         = 5      
MSER_BLOCO_INI = 200_000
MSER_BLOCO_EXT = 100_000

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

# ─── MSER-5Y ────────────────────────────────────────────────────────
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
        samples = sim.samples
        n = len(samples)
        m = n // k
        metade = m // 2
        cinco_pct = max(1, m // 20)
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

        return melhores[0] * k

# ─── Teste de von Neumann ───────────────────────────────────────────
def teste_rank_von_neumann(medias_blocos):
    B = len(medias_blocos)
    # Calcular R_i
    R = [0] * B
    for i in range(B):
        R[i] = sum(1 for x in medias_blocos if x <= medias_blocos[i])
        
    # Estatísticas
    R_barra = sum(R) / B
    
    num = sum((R[j] - R[j+1])**2 for j in range(B - 1))
    den = sum((R[j] - R_barra)**2 for j in range(B))
    
    if den == 0:
        return float('inf')
    return num / den

# ─── Fase de Encontrar M mínimo via NBM ─────────────────────────
def encontrar_m_minimo_nbm(sim, d):
    """
    Cresce M até que as B=20 médias passem no teste de von Neumann (RVN <= BETA)
    """
    M = 50 # tamanho de bloco inicial arbitrário
    while True:
        N_necessario = d + (B_NBM * M)
        if len(sim.samples) < N_necessario:
            sim.gerar(N_necessario - len(sim.samples))
            
        dados = sim.samples[d:d + (B_NBM * M)]
        
        # Calcular as B médias
        medias_blocos = [sum(dados[i*M:(i+1)*M])/M for i in range(B_NBM)]
        
        rvn = teste_rank_von_neumann(medias_blocos)
        
        # Se RVN > Beta, ainda há correlação -> aumenta M
        if rvn <= BETA_CRITICO:
            return M
        M = int(M * 1.2) + 1

# ─── Execução do OBM ──────────────────────
def estimar_com_obm_fonte(sim, d, M_minimo, pct_overlap):
    """
    Aplica a fase de estimação OBM usando a fórmula exata de H do slide.
    """
    M = M_minimo
    while True:
        # Define o deslocamento (shift) com base no overlap solicitado
        if pct_overlap == 1.0:
            c = 1
        elif pct_overlap == 0.50:
            c = max(1, M // 2)
        else:
            c = max(1, int(M * 0.75))
            
        Y_atual = len(sim.samples) - d
        
        dados = sim.samples[d:]
        Y = len(dados)
        
        pref_sum = [0.0] * (Y + 1)
        for i in range(Y):
            pref_sum[i+1] = pref_sum[i] + dados[i]
            
        medias_obm = []
        inicio = 0
        while inicio + M <= Y:
            soma_lote = pref_sum[inicio + M] - pref_sum[inicio]
            medias_obm.append(soma_lote / M)
            inicio += c
            
        B_linha = len(medias_obm) # B' = N - M + 1 (se c=1)
        
        if B_linha < 5:
            sim.gerar(100_000)
            continue
            
        media_global = sum(medias_obm) / B_linha 
        
        # Variância das médias amostrais sobrepostas (s^2_B')
        variancia_lotes = sum((x - media_global)**2 for x in medias_obm) / (B_linha - 1)
        
        # H = t * sqrt((1.5 * M * s^2) / B')
        graus_liberdade = max(2, int((2 * B_linha) / 3)) 
        t_valor = stats.t.ppf(1 - (1 - CONFIDANCA)/2, df=graus_liberdade)
        
        termo_sob_raiz = (1.5 * M * variancia_lotes) / B_linha
        H = t_valor * math.sqrt(max(0.0, termo_sob_raiz))
        
        precisao_relativa = H / media_global if media_global > 0 else float("inf")
        
        # Condição de Parada
        if precisao_relativa <= GAMMA or Y > 10_000_000:
            return {
                "mean": media_global,
                "ic_lo": media_global - H,
                "ic_hi": media_global + H,
                "n_final": d + Y,
                "d": d,
                "m": M,
                "B_linha": B_linha
            }
            
        # Se não atingiu a precisão: Aumentar o valor de M 
        M = int(M * 1.1) + 1
        sim.gerar(M * 10)

# ─── Execução ─────────────────────────────────────────────────────────────────
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

mapeamento_resultados = {ov["rotulo"]: [] for ov in overlaps}

print("=" * 80)
print(" OBM em Conformidade Estrita com as Notas de Aula ICC305")
print("=" * 80)

for ceno in cenarios:
    lam = ceno["lambda"]
    mu = ceno["mu"]
    rho = lam / mu
    wq_teorico = (rho / mu) / (1 - rho)
    
    print(f"\nIniciando Cenário {ceno['id']} (λ={lam}, μ={mu}, ρ={rho:.2f})...")
    
    sim_base = SimuladorMM1(lam, mu)
    sim_base.gerar(MSER_BLOCO_INI)
    
    d_est = mser5_warmup(sim_base)
    print(f"  [1] Transiente Removido (MSER-5Y): d* = {d_est:,}")
    
    # Executa NBM para achar o M estável inicial livre de correlação via teste RVN
    m_estavel_inicial = encontrar_m_minimo_nbm(sim_base, d_est)
    print(f"  [2] Fase NBM concluída: M mínimo estável encontrado = {m_estavel_inicial:,}")
    
    for ov in overlaps:
        t_ini = time.time()
        # Roda a fase de Estimação OBM a partir do M livre de correlação
        res = estimar_com_obm_fonte(sim_base, d_est, m_estavel_inicial, ov["valor"])
        res.update({"teorico": wq_teorico, "id": ceno["id"], "rho": rho})
        mapeamento_resultados[ov["rotulo"]].append(res)
        print(f"     * OBM Overlap {ov['rotulo']}: Concluído em {time.time()-t_ini:.2f}s (N final={res['n_final']:,}, M final={res['m']:,})")

# ─── Geração dos Gráficos ──────────────────────────────────────────
colors = ["#185FA5", "#0F6E56", "#D85A30", "#A32A2A"]

for rotulo, lista_res in mapeamento_resultados.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, res in enumerate(lista_res):
        ax = axes[i]
        ax.bar([0], [res["mean"]], width=0.4, color=colors[i], alpha=0.85, edgecolor=colors[i])
        
        err_lo = max(0, res["mean"] - res["ic_lo"])
        err_hi = res["ic_hi"] - res["mean"]
        ax.errorbar([0], [res["mean"]], yerr=[[err_lo], [err_hi]], fmt="none", color="#2C2C2A", capsize=10, capthick=2, elinewidth=2)
        
        ax.axhline(res["teorico"], color="#E65100", linewidth=2, linestyle="--", label=f"Teórico ({res['teorico']:.4f} s)")
        
        ax.set_title(f"Cenário {res['id']} (ρ = {res['rho']:.2f})\nd*={res['d']:,} | M final={res['m']:,} | B'={res['B_linha']:,}\nN={res['n_final']:,}", fontsize=9, fontweight='bold')
        ax.set_ylabel("Wq (segundos)", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.6)
        
        txt_box = f"Estimado: {res['mean']:.4f} s\nIC: [{res['ic_lo']:.4f}, {res['ic_hi']:.4f}]"
        ax.text(0, res["ic_hi"] + (res["ic_hi"] * 0.05), txt_box, ha="center", va="bottom", fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
        
        ax.set_ylim(res["ic_lo"] * 0.5, res["ic_hi"] * 1.4)
        ax.legend(loc="lower right", fontsize=9)

    nome_arquivo = os.path.join(DIR, f"grafico_obm_overlap_{rotulo.replace('%','')}.png")
    plt.suptitle(f"Fila M/M/1 - OBM ({rotulo} Overlap)", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()
    print(f"\n[Gráfico] Salvo em: {nome_arquivo}")