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

# ─── Parâmetros ───────────────────────────────────────────────────────────────
TAXA_ENTRADA   = 9.5
TAXA_SERVICO   = 10.0
GAMMA          = 0.05

# k = tamanho do lote. k=5 é o valor canônico da literatura (MSER-5).
# Lotes pequenos dão alta resolução à curva MSER(d), essencial para ρ alto
# (ρ=0.95), onde a curva é quase monotônica e o mínimo real fica em d baixo.
# k=500 gera lotes grandes demais: com poucos lotes o mínimo aparece no último
# da 1ª metade (~50% das amostras descartadas), causando viés severo para baixo.
MSER_K         = 5

MSER_BLOCO_INI = 2_000_000  # 2M amostras → 400k lotes com k=5 (resolução alta)
MSER_BLOCO_EXT = 500_000    # expansão em caso de re-tentativa

IC_FIRST_CHECK = 5_000
IC_RECALC      = 1_000



rho    = TAXA_ENTRADA / TAXA_SERVICO
Wq_teo = (rho / TAXA_SERVICO) / (1 - rho)

DIR = os.path.dirname(os.path.abspath(__file__))

CSV_CONWAY  = os.path.join(DIR, "resultados_resumo5_Conway.csv")
CSV_FISHMAN = os.path.join(DIR, "resultados_resumo5_Fishman.csv")
CSV_MSER    = os.path.join(DIR, "resultados_resumo5_MSER5.csv")
PNG_SAIDA   = os.path.join(DIR, "grafico_mm1_heuristicas.png")

# ─── Leitura dos CSVs ─────────────────────────────────────────────────────────
def ler_csv(caminho):
    linhas = []
    with open(caminho, newline="") as f:
        for row in csv.DictReader(f):
            if str(row["n_iteracao"]).strip().lower().startswith("m"):
                continue
            linhas.append(row)
    return linhas

def resumo_csv(linhas):
    medias = [float(r["media_estimada"]) for r in linhas]
    ic_los = [float(r["ic_inferior"])    for r in linhas]
    ic_his = [float(r["ic_superior"])    for r in linhas]
    no_ic  = [r["teorico_no_ic"].strip().lower() == "true" for r in linhas]
    n      = len(medias)
    return dict(
        mean     = sum(medias) / n,
        ic_lo    = sum(ic_los) / n,
        ic_hi    = sum(ic_his) / n,
        cobertura= sum(no_ic)  / n,
        n_iter   = n,
    )

# ─── Simulador M/M/1 ──────────────────────────────────────────────────────────
class SimuladorMM1:
    def __init__(self):
        self.t        = 0.0
        self.queue    = 0
        self.busy     = False
        self.arrivals = deque()   # O(1) popleft
        self.next_arr = random.expovariate(TAXA_ENTRADA)
        self.next_dep = float("inf")
        self.samples  = []

    def gerar(self, n, verbose=False):
        alvo     = len(self.samples) + n
        marco    = len(self.samples) + max(n // 4, 1)
        t_inicio = time.time()
        while len(self.samples) < alvo:
            if self.next_arr <= self.next_dep:
                self.t = self.next_arr
                if not self.busy:
                    self.busy     = True
                    self.next_dep = self.t + random.expovariate(TAXA_SERVICO)
                    self.samples.append(0.0)
                else:
                    self.queue += 1
                    self.arrivals.append(self.t)
                self.next_arr = self.t + random.expovariate(TAXA_ENTRADA)
            else:
                self.t = self.next_dep
                if self.queue > 0:
                    self.queue   -= 1
                    wq            = self.t - self.arrivals.popleft()
                    self.samples.append(wq)
                    self.next_dep = self.t + random.expovariate(TAXA_SERVICO)
                else:
                    self.busy     = False
                    self.next_dep = float("inf")
            if verbose and len(self.samples) >= marco:
                pct = (len(self.samples) - (alvo - n)) / n * 100
                el  = time.time() - t_inicio
                print(f"    {pct:5.1f}%  ({len(self.samples):,} amostras, {el:.1f}s)")
                marco += max(n // 4, 1)

# ─── MSER-5Y — cálculo via somas de sufixo ────────────────────────────────────
def _mser_sufixo(batches, d_ini, d_fim):
    """Calcula MSER(d) para d em [d_ini, d_fim) em O(m) via somas de sufixo."""
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

# ─── MSER-5Y — implementação fiel à literatura ────────────────────────────────
def mser5_warmup(sim, k=MSER_K):
    """
    MSER-5Y conforme a literatura (Banks et al., Law & Kelton):
      1. Dividir N observações em lotes de tamanho k → m = N/k lotes Z_j.
      2. Calcular MSER(d) para d = 0 .. m//2  (primeira metade).
      3. Escolher d* = argmin MSER(d) na primeira metade.
      4. Re-tentativa em dois casos especiais:
         a) Empate: dois ou mais d com o mesmo valor mínimo de MSER(d)
            na primeira metade.
         b) Invasão: existe algum d na faixa (m//2, m//2 + m//20]
            (os 5% iniciais da segunda metade) com MSER(d) menor que
            o mínimo encontrado na primeira metade.
         Em ambos os casos: coletar mais amostras e reiniciar todo o
         procedimento.
    """
    tentativa = 0
    while True:
        samples    = sim.samples
        n          = len(samples)
        m          = n // k
        metade     = m // 2
        cinco_pct  = max(1, m // 20)       # 5% dos lotes totais
        fim_janela = metade + cinco_pct    # índice final da janela na 2ª metade

        if m < 20:
            print(f"  MSER-5Y: poucos lotes ({m}), gerando mais...")
            sim.gerar(MSER_BLOCO_EXT, verbose=True)
            tentativa += 1
            continue

        t0 = time.time()
        batches = [sum(samples[b*k:(b+1)*k]) / k for b in range(m)]

        # Calcula MSER da 1ª metade e dos 5% iniciais da 2ª metade
        mser1        = _mser_sufixo(batches, 0,      metade)
        mser2_janela = _mser_sufixo(batches, metade, fim_janela)
        t_calc       = time.time() - t0

        best_mser = min(mser1, key=lambda x: x[0])[0]
        melhores  = [d for mser, d in mser1 if mser == best_mser]

        # Menor MSER encontrado nos 5% iniciais da 2ª metade
        min_mser2 = min(
            (mser for mser, d in mser2_janela),
            default=float("inf")
        )

        print(f"  MSER calculado: {t_calc:.3f}s  "
              f"m={m:,} lotes  metade={metade:,}  "
              f"janela_2a=[{metade},{fim_janela})  "
              f"n_min={len(melhores)} candidato(s)  "
              f"best_mser1={best_mser:.6f}  min_mser2={min_mser2:.6f}")

        # Caso 1: empate na 1ª metade
        if len(melhores) > 1:
            tentativa += 1
            print(f"  MSER-5Y tentativa {tentativa}: empate em {len(melhores)} "
                  f"lotes → +{MSER_BLOCO_EXT:,} amostras")
            sim.gerar(MSER_BLOCO_EXT, verbose=True)
            continue

        # Caso 2: invasão — 5% iniciais da 2ª metade têm valor menor
        if min_mser2 < best_mser:
            tentativa += 1
            print(f"  MSER-5Y tentativa {tentativa}: invasão da 2ª metade "
                  f"(min_mser2={min_mser2:.6f} < best_mser1={best_mser:.6f}) "
                  f"→ +{MSER_BLOCO_EXT:,} amostras")
            sim.gerar(MSER_BLOCO_EXT, verbose=True)
            continue

        # d* aceito: único mínimo na 1ª metade e sem invasão
        best_d = melhores[0]
        d_star = best_d * k
        print(f"  MSER-5Y convergiu em {tentativa} re-tentativa(s): "
              f"d*={d_star:,}  (lote {best_d}/{metade}, mser={best_mser:.6f})")
        return d_star

# ─── Estimação com regra de parada H/X̄ ≤ γ ───────────────────────────────────
def estimate_with_stop(sim, d):
    mean = 0.0
    M2   = 0.0
    n    = 0
    idx  = d

    while True:
        if idx >= len(sim.samples):
            sim.gerar(IC_RECALC * 10)

        v     = sim.samples[idx]
        idx  += 1
        n    += 1
        delta = v - mean
        mean += delta / n
        M2   += delta * (v - mean)

        if n == IC_FIRST_CHECK or (n > IC_FIRST_CHECK and
                                    (n - IC_FIRST_CHECK) % IC_RECALC == 0):
            if n > 1:
                sd = math.sqrt(M2 / (n - 1))
                H  = 1.96 * sd / math.sqrt(n)
                if mean > 0 and H / mean <= GAMMA:
                    break

    sd    = math.sqrt(M2 / (n - 1))
    H     = 1.96 * sd / math.sqrt(n)
    ic_lo = mean - H
    ic_hi = mean + H
    ratio = H / mean if mean > 0 else float("inf")
    return dict(mean=mean, sd=sd, ic_lo=ic_lo, ic_hi=ic_hi,
                H=H, ratio=ratio, n=n, d=d,
                n_total=len(sim.samples))

# ─── Grava CSV ────────────────────────────────────────────────────────────────
def gravar_csv_mser(resultados, duracoes):
    cabecalho = ["n_iteracao","n_alvo","n_amostrado","lambda","mu","rho",
                 "media_estimada","wq_teorico","vies","desvio_padrao",
                 "ic_inferior","ic_superior","teorico_no_ic","duracao_wall_s"]
    with open(CSV_MSER, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cabecalho)
        for i, (res, dur) in enumerate(zip(resultados, duracoes), start=1):
            no_ic = res["ic_lo"] <= Wq_teo <= res["ic_hi"]
            vies  = res["mean"] - Wq_teo
            w.writerow([i, res["n"], res["n_total"],
                        TAXA_ENTRADA, TAXA_SERVICO, round(rho, 6),
                        round(res["mean"],  8), round(Wq_teo, 8),
                        round(vies,         8), round(res["sd"],   8),
                        round(res["ic_lo"], 8), round(res["ic_hi"], 8),
                        no_ic, round(dur, 3)])

# ─── Gráfico ──────────────────────────────────────────────────────────────────
def plotar(res_c, res_f, res_m):
    heuristicas = [
        ("Conway",  res_c, "#185FA5"),
        ("Fishman", res_f, "#0F6E56"),
        ("MSER-5Y", res_m, "#D85A30"),
    ]
    means  = [h[1]["mean"]  for h in heuristicas]
    ic_los = [h[1]["ic_lo"] for h in heuristicas]
    ic_his = [h[1]["ic_hi"] for h in heuristicas]
    colors = [h[2]          for h in heuristicas]
    x_pos  = [0, 1, 2]
    err_lo = [m - l for m, l in zip(means, ic_los)]
    err_hi = [h - m for m, h in zip(means, ic_his)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos, means, width=0.45, color=colors, alpha=0.80,
           edgecolor=colors, linewidth=1.2, zorder=3)
    ax.errorbar(x_pos, means, yerr=[err_lo, err_hi],
                fmt="none", color="#2C2C2A",
                capsize=7, capthick=1.5, elinewidth=1.5, zorder=4)
    ax.axhline(Wq_teo, color="#888780", linewidth=1.8, linestyle="--", zorder=2)

    for xi, (name, res, _) in zip(x_pos, heuristicas):
        cobert = res.get("cobertura", int(res["ic_lo"] <= Wq_teo <= res["ic_hi"]))
        ax.text(xi, res["ic_hi"] + 0.02,
                f"{res['mean']:.4f} s\ncob={cobert*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#2C2C2A")

    margem = (max(ic_his) - min(ic_los)) * 0.4
    ax.set_ylim(min(ic_los) - margem, max(ic_his) + margem * 3.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([h[0] for h in heuristicas], fontsize=12)
    ax.set_ylabel("Wq  (segundos)", fontsize=11)
    ax.set_title(
        f"M/M/1 — λ={TAXA_ENTRADA}  μ={TAXA_SERVICO}  ρ={rho:.2f} — "
        "Comparação de heurísticas de remoção de transiente",
        fontsize=11, pad=12)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [mpatches.Patch(color=c, label=n, alpha=0.8)
               for n, _, c in heuristicas]
    patches.append(plt.Line2D([0], [0], color="#888780", linewidth=1.8,
                              linestyle="--",
                              label=f"Teórico E[Wq] = {Wq_teo:.4f} s"))
    ax.legend(handles=patches, fontsize=9, loc="upper right", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(PNG_SAIDA, dpi=150)
    plt.close()
    print(f"  Gráfico salvo em: {PNG_SAIDA}")

# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("    SIMULAÇÃO M/M/1  —  λ=9.5  μ=10  ρ=0.95")
print(f"    E[Wq] teórico = {Wq_teo:.6f} s")
print(f"    Regra de parada: H/X̄(n) ≤ γ = {GAMMA} ({GAMMA*100:.0f}%)")
print("=" * 65)

print("\n[1/3] Lendo CSVs de Conway e Fishman...")
res_c = resumo_csv(ler_csv(CSV_CONWAY))
res_f = resumo_csv(ler_csv(CSV_FISHMAN))
print(f"  Conway  → Ŵq={res_c['mean']:.4f}  "
      f"IC=[{res_c['ic_lo']:.4f},{res_c['ic_hi']:.4f}]  "
      f"cobertura={res_c['cobertura']*100:.1f}%")
print(f"  Fishman → Ŵq={res_f['mean']:.4f}  "
      f"IC=[{res_f['ic_lo']:.4f},{res_f['ic_hi']:.4f}]  "
      f"cobertura={res_f['cobertura']*100:.1f}%")

print(f"\n[2/3] Simulação MSER-5Y (horizonte infinito, γ={GAMMA})...")
t0  = time.time()
sim = SimuladorMM1()

print(f"  Gerando bloco inicial de {MSER_BLOCO_INI:,} amostras...")
sim.gerar(MSER_BLOCO_INI, verbose=True)
d = mser5_warmup(sim)

print(f"  Estimando Wq a partir de d*={d:,} "
      f"(1ª verificação em {IC_FIRST_CHECK:,}, depois a cada {IC_RECALC:,})...")
res_m_raw = estimate_with_stop(sim, d)
duracao   = time.time() - t0

no_ic = res_m_raw["ic_lo"] <= Wq_teo <= res_m_raw["ic_hi"]
print(f"  MSER-5Y → Ŵq={res_m_raw['mean']:.4f}  "
      f"IC=[{res_m_raw['ic_lo']:.4f},{res_m_raw['ic_hi']:.4f}]  "
      f"H/X̄={res_m_raw['ratio']*100:.2f}%  "
      f"{'✓' if no_ic else '✗'}  ({duracao:.1f}s)")

gravar_csv_mser([res_m_raw], [duracao])
print(f"  CSV salvo em: {CSV_MSER}")

res_m = dict(
    mean     = res_m_raw["mean"],
    ic_lo    = res_m_raw["ic_lo"],
    ic_hi    = res_m_raw["ic_hi"],
    cobertura= int(no_ic),
)

print("\n[3/3] Gerando gráfico comparativo...")
plotar(res_c, res_f, res_m)

print(f"\n  Tempo total: {duracao:.1f}s")
print("=" * 65)