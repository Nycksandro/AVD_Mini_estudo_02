import math
import random
import time
import os
from collections import deque
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Parâmetros Gerais ──────────────────────────────────────
GAMMA        = 0.05   
B_NBM        = 20   
BETA_CRITICO = 1.44   
CONFIDANCA   = 0.95
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

# ─── MSER-5Y ─────────────────────────────────────────────────────
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

# ─── Estatística de Rank von Neumann ────────────────────────────────
def teste_rank_von_neumann(medias_blocos):
    B = len(medias_blocos)
    R = [0] * B
    for i in range(B):
        R[i] = sum(1 for x in medias_blocos if x <= medias_blocos[i]) 
        
    R_barra = sum(R) / B # Média dos postos 
    
    num = sum((R[j] - R[j+1])**2 for j in range(B - 1))
    den = sum((R[j] - R_barra)**2 for j in range(B))
    
    if den == 0:
        return float('inf')
    return num / den

# ─── IMPLEMENTAÇÃO DO NBM────────────────────
def executar_nbm_estrito(sim, d):
    M = 100 # tamanho inicial do lote
    while True:
        N_necessario = d + (B_NBM * M)
        if len(sim.samples) < N_necessario:
            sim.gerar(N_necessario - len(sim.samples))
            
        dados_estacionarios = sim.samples[d:d + (B_NBM * M)]
        medias_blocos = [sum(dados_estacionarios[i*M:(i+1)*M])/M for i in range(B_NBM)]
        
        rvn = teste_rank_von_neumann(medias_blocos)
        
        # Se RVN > Beta, ainda há correlação significativa
        if rvn > BETA_CRITICO:
            M = int(M * 1.2) + 1 # Incrementa M de forma geométrica para evitar loops infinitos
            continue
            
        # Fase de Estimação NBM Clássica
        media_global = sum(medias_blocos) / B_NBM
        variancia_blocos = sum((x - media_global)**2 for x in medias_blocos) / (B_NBM - 1)
        erro_padrao = math.sqrt(variancia_blocos / B_NBM)
        
        # IC t-Student clássico com B-1 graus de liberdade
        t_valor = stats.t.ppf(1 - (1 - CONFIDANCA)/2, df=B_NBM - 1)
        H = t_valor * erro_padrao
        
        precisao_relativa = H / media_global if media_global > 0 else float("inf")
        
        # Critério de parada de precisão de 5% 
        if precisao_relativa <= GAMMA or (B_NBM * M) > 12_000_000:
            return {
                "mean": media_global, "ic_lo": media_global - H, "ic_hi": media_global + H,
                "n_final": d + (B_NBM * M), "d": d, "m": M, "B": B_NBM
            }
        
        M = int(M * 1.1) + 1

# ─── IMPLEMENTAÇÃO DO OBM ────────────────────
def executar_obm_estrito(sim, d, pct_overlap):
    M = 1000  # Tamanho de lote base. O OBM cresce M para varrer a correlação assintoticamente
    while True:
        # Define o salto geométrico de avanço conforme o grau de overlap
        if pct_overlap == 1.0:
            c = 1
        elif pct_overlap == 0.50:
            c = max(1, M // 2)
        else:
            c = max(1, int(M * 0.75))
            
        Y = len(sim.samples) - d
        if Y < M * 50: # Garante dados suficientes para computar lotes sobrepostos significativos
            sim.gerar(M * 50)
            Y = len(sim.samples) - d
            
        dados = sim.samples[d:]
        
        # Otimização por somas acumuladas (Prefix Sums) para garantir velocidade em O(N)
        pref_sum = [0.0] * (Y + 1)
        for i in range(Y):
            pref_sum[i+1] = pref_sum[i] + dados[i]
            
        medias_obm = []
        inicio = 0
        while inicio + M <= Y:
            soma_lote = pref_sum[inicio + M] - pref_sum[inicio]
            medias_obm.append(soma_lote / M)
            inicio += c
            
        B_linha = len(medias_obm) # Número total de lotes sobrepostos gerados (B')
        if B_linha < 10:
            M = int(M * 1.1) + 1
            continue
            
        media_global = sum(medias_obm) / B_linha
        variancia_lotes = sum((x - media_global)**2 for x in medias_obm) / (B_linha - 1)
        
        # EQUAÇÃO EXATA DO SLIDE DE OBM: H baseado em 1.5 * n - 1 graus de liberdade aproximados
        gl_ajustado = max(2, int((2 * B_linha) / 3)) 
        t_valor = stats.t.ppf(1 - (1 - CONFIDANCA)/2, df=gl_ajustado)
        
        # Termo sob a raiz explícito H = t * sqrt( (1.5 * M * s^2) / B' )
        termo_raiz = (1.5 * M * variancia_lotes) / B_linha
        H = t_valor * math.sqrt(max(0.0, termo_raiz))
        
        precisao_relativa = H / media_global if media_global > 0 else float("inf")
        
        if precisao_relativa <= GAMMA or Y > 12_000_000:
            return {
                "mean": media_global, "ic_lo": media_global - H, "ic_hi": media_global + H,
                "n_final": d + Y, "d": d, "m": M, "B_linha": B_linha
            }
            
        M = int(M * 1.1) + 1
        sim.gerar(M * 5)

# ─── Execução e Validação dos Cenários ────────────────────────────────────────
cenarios = [
    {"id": "I",   "lambda": 7.0,  "mu": 10.0},
    {"id": "II",  "lambda": 8.0,  "mu": 10.0},
    {"id": "III", "lambda": 9.0,  "mu": 10.0},
    {"id": "IV",  "lambda": 9.5,  "mu": 10.0}
]

print("=" * 80)
print(" Execução")
print("=" * 80)

for ceno in cenarios:
    lam = ceno["lambda"]
    mu = ceno["mu"]
    rho = lam / mu
    wq_teorico = (rho / mu) / (1 - rho)
    
    print(f"\n▶ Cenário {ceno['id']} (λ={lam}, μ={mu}, ρ={rho:.2f})")
    
    sim_base = SimuladorMM1(lam, mu)
    sim_base.gerar(MSER_BLOCO_INI)
    d_est = mser5_warmup(sim_base)
    print(f"   [Warmup MSER-5Y]: Transiente d* = {d_est:,}")
    
    # Teste e Estimação via NBM Puro
    t0 = time.time()
    res_nbm = executar_nbm_estrito(sim_base, d_est)
    print(f"   [NBM Puro] Concluído em {time.time()-t0:.2f}s | N={res_nbm['n_final']:,} | M={res_nbm['m']:,} | Wq={res_nbm['mean']:.4f}")
    
    # Estimação via OBM Puro (Exemplo com Overlap de 100%)
    t0 = time.time()
    res_obm = executar_obm_estrito(sim_base, d_est, 1.0)
    print(f"   [OBM 100%] Concluído em {time.time()-t0:.2f}s | N={res_obm['n_final']:,} | M={res_obm['m']:,} | Wq={res_obm['mean']:.4f}")

print("\nConcluído!")