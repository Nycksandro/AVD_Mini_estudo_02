import csv
import math
import random
import time
import os

# parametros gerais
TAXA_ENTRADA = 9
TAXA_SERVICO = 10
GAMA         = 0.05   # precisão relativa γ = 5%
RECALC_CADA  = 50    # verifica condição a cada N amostras
CSV_RESUMO   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_resumo3.csv")

rho    = TAXA_ENTRADA / TAXA_SERVICO
Wq_teo = (rho / TAXA_SERVICO) / (1 - rho)

with open(CSV_RESUMO, "w", newline="") as f:
    csv.writer(f).writerow([
        "gamma", "n_final", "lambda", "mu", "rho",
        "media_X_n", "wq_teorico",
        "desvio_padrao", "ic_inferior", "ic_superior",
        "H", "H_sobre_X", "teorico_no_ic", "duracao_wall_s"
    ])

print("─" * 60)
print("         SIMULAÇÃO M/M/1  –  λ=9  μ=10")
print("         Regra de parada: tamanho relativo do IC")
print(f"         Para quando H / X̄(n) ≤ γ = {GAMA} ({GAMA*100:.0f}%)")
print(f"         Verifica a cada {RECALC_CADA} amostras")
print("─" * 60)

# reinicio de variaveis
tempo_atual      = 0.0
fila             = 0
servidor_ocupado = False
fila_chegada     = []
proxima_chegada  = random.expovariate(TAXA_ENTRADA)
proxima_saida    = float('inf')

wf_n    = 0
wf_mean = 0.0
wf_M2   = 0.0

H            = float('inf')
H_sobre_X    = float('inf')
inicio_wall  = time.time()

# loop para condição de parada
while H_sobre_X > GAMA:

    if proxima_chegada <= proxima_saida:
        # chegada
        tempo_atual = proxima_chegada
        if not servidor_ocupado:
            wf_n += 1
            delta    = 0.0 - wf_mean
            wf_mean += delta / wf_n
            wf_M2   += delta * (0.0 - wf_mean)
            servidor_ocupado = True
            proxima_saida    = tempo_atual + random.expovariate(TAXA_SERVICO)
        else:
            fila += 1
            fila_chegada.append(tempo_atual)
        proxima_chegada = tempo_atual + random.expovariate(TAXA_ENTRADA)

    else:
        # saida
        tempo_atual = proxima_saida
        if fila > 0:
            fila -= 1
            wq = tempo_atual - fila_chegada.pop(0)
            wf_n += 1
            delta    = wq - wf_mean
            wf_mean += delta / wf_n
            wf_M2   += delta * (wq - wf_mean)
            proxima_saida = tempo_atual + random.expovariate(TAXA_SERVICO)
        else:
            servidor_ocupado = False
            proxima_saida    = float('inf')

    # verifica condicao a cada RECALC_CADA amostras
    if wf_n >= 2 and wf_n % RECALC_CADA == 0:
        desvio    = math.sqrt(wf_M2 / (wf_n - 1))
        H         = 1.96 * desvio / math.sqrt(wf_n)   # meia-largura
        if wf_mean > 0:
            H_sobre_X = H / wf_mean                   # precisão relativa

# estatisticas
duracao_wall = time.time() - inicio_wall
n      = wf_n
media  = wf_mean
desvio = math.sqrt(wf_M2 / (n - 1))
H      = 1.96 * desvio / math.sqrt(n)
ic_inf = media - H
ic_sup = media + H
no_ic  = ic_inf <= Wq_teo <= ic_sup
H_sobre_X = H / media

with open(CSV_RESUMO, "a", newline="") as f:
    csv.writer(f).writerow([
        GAMA, n,
        TAXA_ENTRADA, TAXA_SERVICO, round(rho, 6),
        round(media,  8), round(Wq_teo, 8),
        round(desvio, 8), round(ic_inf, 8), round(ic_sup, 8),
        round(H, 8), round(H_sobre_X, 6), no_ic, round(duracao_wall, 3)
    ])

print(f"\n{'='*60}")
print(f"  γ = {GAMA}   →   parou com H/X̄(n) ≤ {GAMA}")
print(f"{'='*60}")
print(f"  n final                      : {n:,}")
print(f"  Média estimada  X̄(n)         : {media:.6f} s")
print(f"  Valor teórico   E[Wq]        : {Wq_teo:.6f} s")
print(f"  Desvio-padrão               : {desvio:.6f} s")
print(f"  H (meia-largura IC)          : {H:.6f}")
print(f"  H / X̄(n)                    : {H_sobre_X:.6f}  (alvo: ≤ {GAMA})")
print(f"  Intervalo de confiança (95%) : [{ic_inf:.6f} , {ic_sup:.6f}]")
print(f"  Teórico no IC               : {'✓ Sim' if no_ic else '✗ Não'}")
print(f"  Wall-clock                  : {duracao_wall:.2f}s")
print("\n" + "─" * 60)
print(f"  CSV salvo em: {CSV_RESUMO}")
print("─" * 60)