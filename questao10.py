"""
Exercício 10 — Simulação M/M/1 de horizonte infinito
  • Remoção de transiente : MSER-5Y
  • Estimação             : STS/AREA (procedimento fiel à literatura)
  • Regra de parada       : H/X̄ ≤ ε = 5 %  (IC 95 %)

Procedimento STS/AREA conforme slides:
  Passos 1-4 : Selecionar B e M, coletar N=B×M obs., computar Aᵢ, testar normalidade (SW)
  Passo 5    : Se normalidade falha → aumentar M (e B, para reduzir autocorr) → volta passo 2
  Passo 6    : Calcular X̄(B) e H_A
  Passo 7    : Se H_A/X̄(B) ≥ ε → aumentar M → recoletar → calcular Aᵢ → volta passo 6
               (NÃO testa mais normalidade)
  Passo 8    : Caso contrário, terminar
"""

import math, random, time
from collections import deque

import numpy as np
from scipy.stats import shapiro, t as t_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Parâmetros ───────────────────────────────────────────────────────────────
MU       = 10.0
EPSILON  = 0.05      # precisão relativa 5 %
ALPHA_SW = 0.05      # nível Shapiro-Wilk
CENARIOS = [7.0, 8.0, 9.0, 9.5]

B_INI  = 500    # tamanho inicial de cada bloco
B_MULT = 2      # fator de aumento de B quando normalidade falha
M_INI  = 30     # blocos iniciais
M_INC  = 10     # incremento de M quando IC não fecha (Fase B)

MSER_K     = 5
MSER_BLOCO = 1_000_000
MSER_EXT   = 300_000

# ─── Simulador M/M/1 ─────────────────────────────────────────────────────────
class SimuladorMM1:
    def __init__(self, lam, mu):
        self.lam=lam; self.mu=mu; self.t=0.0
        self.queue=0; self.busy=False
        self.arrivals=deque()
        self.next_arr=random.expovariate(lam)
        self.next_dep=float("inf")
        self.samples=[]

    def gerar(self, n):
        alvo=len(self.samples)+n
        while len(self.samples)<alvo:
            if self.next_arr<=self.next_dep:
                self.t=self.next_arr
                if not self.busy:
                    self.busy=True
                    self.next_dep=self.t+random.expovariate(self.mu)
                    self.samples.append(0.0)
                else:
                    self.queue+=1; self.arrivals.append(self.t)
                self.next_arr=self.t+random.expovariate(self.lam)
            else:
                self.t=self.next_dep
                if self.queue>0:
                    self.queue-=1
                    self.samples.append(self.t-self.arrivals.popleft())
                    self.next_dep=self.t+random.expovariate(self.mu)
                else:
                    self.busy=False; self.next_dep=float("inf")

# ─── MSER-5Y ─────────────────────────────────────────────────────────────────
def _sufixo(batches, d0, d1):
    M=len(batches); sx=[0.0]*(M+1); sx2=[0.0]*(M+1)
    for i in range(M-1,-1,-1):
        sx[i]=sx[i+1]+batches[i]; sx2[i]=sx2[i+1]+batches[i]**2
    out=[]
    for d in range(d0,min(d1,M-2)):
        c=M-d; mu_d=sx[d]/c; out.append((sx2[d]/c-mu_d*mu_d,d))
    return out

def mser5_warmup(sim):
    while True:
        if len(sim.samples)<MSER_BLOCO: sim.gerar(MSER_BLOCO-len(sim.samples))
        k=MSER_K; m=len(sim.samples)//k; half=m//2; w=max(1,m//20)
        if m<20: sim.gerar(MSER_EXT); continue
        b=[sum(sim.samples[j*k:(j+1)*k])/k for j in range(m)]
        s1=_sufixo(b,0,half); s2=_sufixo(b,half,half+w)
        best=min(s1,key=lambda x:x[0])[0]
        bests=[d for v,d in s1 if v==best]
        min2=min((v for v,_ in s2),default=float("inf"))
        if len(bests)>1 or min2<best: sim.gerar(MSER_EXT); continue
        return bests[0]*k

# ─── STS/AREA ────────────────────────────────────────────────────────────────
def sts_area(sim, d):
    """
    Fase A — Teste de normalidade (passos 1-5):
      Quando SW falha → dobra B (blocos maiores quebram autocorrelação)
      e reinicia M = M_INI para manter a amostra mínima.
      (Aumentar só M com B fixo não ajuda quando autocorr é o problema.)

    Fase B — Critério de parada por IC (passos 6-7):
      B fica FIXO (valor aceito na Fase A).
      Quando H/X̄ ≥ ε → aumenta M e recalcula sem testar SW novamente.
    """
    B = B_INI
    M = M_INI

    # ══════════════════════════════════════════════════════
    # FASE A: Testar normalidade dos Aᵢ (passos 1-5)
    # ══════════════════════════════════════════════════════
    print(f"        [Fase A] Testando normalidade dos Aᵢ...")
    while True:
        # Passo 2: coletar N = B×M observações
        N = B * M
        total = d + N
        if len(sim.samples) < total:
            sim.gerar(total - len(sim.samples))

        dados = np.array(sim.samples[d:d+N], dtype=np.float64)
        grand = dados.mean()

        # Passo 3: computar Aᵢ para cada bloco
        Z = dados.reshape(M, B).mean(axis=1)
        A = Z - grand

        # Passo 4: Shapiro-Wilk (limitado a 5000 pela scipy)
        amostra_sw = A[:5000] if M > 5000 else A
        _, p_sw = shapiro(amostra_sw)
        ok_sw = p_sw >= ALPHA_SW
        print(f"        Shapiro-Wilk: p={p_sw:.4f}  "
              f"({'✓ normal' if ok_sw else '✗ rejeitada'})"
              f"  B={B}  M={M}  N={N:,}")

        if ok_sw:
            break  # normalidade aceita → vai para Fase B

        # Passo 5: normalidade falhou → dobra B para reduzir autocorrelação,
        # reinicia M = M_INI e volta ao passo 2
        B *= B_MULT
        M  = M_INI

    # B fica fixo para toda a Fase B
    B_fixo = B

    # ══════════════════════════════════════════════════════
    # FASE B: Critério de parada por IC (passos 6-7)
    # Atenção: NÃO testa mais normalidade (nota do slide)
    # ══════════════════════════════════════════════════════
    print(f"        [Fase B] Critério de parada (B={B_fixo} fixo, sem SW)...")
    while True:
        # Passo 6: computar X̄(B) e H_A com M atual
        N = B_fixo * M
        total = d + N
        if len(sim.samples) < total:
            sim.gerar(total - len(sim.samples))

        dados = np.array(sim.samples[d:d+N], dtype=np.float64)
        grand = dados.mean()
        Z = dados.reshape(M, B_fixo).mean(axis=1)
        A = Z - grand

        var_xbar = np.sum(A**2) / (M * (M - 1))
        H = t_dist.ppf(1 - ALPHA_SW / 2, df=M - 1) * math.sqrt(var_xbar)
        ratio = H / grand if grand > 0 else float("inf")

        print(f"        IC: X̄={grand:.6f}  H={H:.6f}  "
              f"H/X̄={ratio*100:.2f}%  B={B_fixo}  M={M}  N={N:,}")

        # Passo 7: H_A/X̄(B) ≥ ε?
        if ratio >= EPSILON:
            M += M_INC  # aumenta M, recalcula Aᵢ, volta passo 6 (sem SW)
            continue

        # Passo 8: terminar
        break

    return dict(mean=grand, ic_lo=grand-H, ic_hi=grand+H,
                H=H, ratio=ratio, M=M, B=B_fixo, N=N)

# ─── Teórico ─────────────────────────────────────────────────────────────────
def wq_teo(lam, mu):
    rho = lam / mu
    return rho / (mu * (1 - rho))

# ─── Simulações ──────────────────────────────────────────────────────────────
random.seed(42)
resultados = []

for lam in CENARIOS:
    rho = lam / MU
    wteo = wq_teo(lam, MU)
    print(f"\n{'='*62}")
    print(f"  Cenário λ={lam}  μ={MU}  ρ={rho:.2f}  E[Wq]={wteo:.6f} s")
    print(f"{'='*62}")
    t0 = time.time()
    sim = SimuladorMM1(lam, MU)
    print(f"  [1/3] Gerando {MSER_BLOCO:,} amostras iniciais...")
    sim.gerar(MSER_BLOCO)
    print(f"  [2/3] MSER-5Y...")
    d = mser5_warmup(sim)
    print(f"        d* = {d:,} amostras descartadas")
    print(f"  [3/3] STS/AREA  (ε={int(EPSILON*100)}%  α_SW={int(ALPHA_SW*100)}%)...")
    res = sts_area(sim, d)
    dur = time.time() - t0
    no_ic = res["ic_lo"] <= wteo <= res["ic_hi"]
    print(f"\n  ► Ŵq  = {res['mean']:.6f} s")
    print(f"  ► IC  = [{res['ic_lo']:.6f}, {res['ic_hi']:.6f}]")
    print(f"  ► H/X̄= {res['ratio']*100:.2f}%  "
          f"({'✓ teórico no IC' if no_ic else '✗ teórico fora do IC'})")
    print(f"  ► M={res['M']}  B={res['B']}  N={res['N']:,}  ({dur:.1f}s)")
    resultados.append(dict(lam=lam, rho=rho, wteo=wteo, d=d, dur=dur, no_ic=no_ic, **res))

# ─── Gráfico ─────────────────────────────────────────────────────────────────
CORES  = ["#1A6FA8", "#1E9E6B", "#D4790A", "#C0392B"]
LABELS = ["Cenário I\nλ=7", "Cenário II\nλ=8", "Cenário III\nλ=9", "Cenário IV\nλ=9.5"]

fig, ax = plt.subplots(figsize=(11, 6.5))
x = np.arange(len(resultados)); W = 0.5

ax.bar(x, [r["mean"] for r in resultados], width=W,
       color=CORES, alpha=0.82, edgecolor=CORES, linewidth=1.3, zorder=3)
ax.errorbar(x, [r["mean"] for r in resultados],
            yerr=[[r["mean"]-r["ic_lo"] for r in resultados],
                  [r["ic_hi"]-r["mean"] for r in resultados]],
            fmt="none", color="#222", capsize=8, capthick=1.8, elinewidth=1.8, zorder=4)

for xi, r, cor in zip(x, resultados, CORES):
    ax.hlines(r["wteo"], xi-W/2-.06, xi+W/2+.06,
              colors=cor, linewidths=2.5, linestyles="--", zorder=5)

for xi, r in zip(x, resultados):
    marca = "✓" if r["no_ic"] else "✗"
    txt = (f"Ŵq = {r['mean']:.4f}\n"
           f"IC=[{r['ic_lo']:.4f},{r['ic_hi']:.4f}]\n"
           f"Teórico={r['wteo']:.4f} {marca}\n"
           f"d*={r['d']:,}  M={r['M']}  B={r['B']}")
    ax.text(xi, r["ic_hi"]+r["mean"]*.03, txt,
            ha="center", va="bottom", fontsize=7.5, linespacing=1.45)

top = max(r["ic_hi"] for r in resultados)
ax.set_ylim(0, top*2.8)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel("E[Wq]  (segundos)", fontsize=12)
ax.set_title(
    "M/M/1 — Simulação de horizonte infinito\n"
    "Transiente: MSER-5Y  |  Estimação: STS/AREA  |  "
    f"Shapiro-Wilk α={int(ALPHA_SW*100)}%  |  ε={int(EPSILON*100)}%  (IC 95%)",
    fontsize=11, pad=14)
ax.grid(axis="y", color="#DADAD4", linewidth=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

patches = [mpatches.Patch(color=c, alpha=0.82,
           label=f"λ={r['lam']} (ρ={r['rho']:.2f})")
           for c, r in zip(CORES, resultados)]
patches.append(plt.Line2D([0], [0], color="gray", linewidth=2,
               linestyle="--", label="Valor teórico E[Wq]"))
ax.legend(handles=patches, fontsize=9.5, loc="upper left", framealpha=0.85)
plt.tight_layout()
plt.savefig("./ex10_mm1_sts_area.png", dpi=160)
plt.close()
print("\nGráfico salvo em: ex10_mm1_sts_area.png")