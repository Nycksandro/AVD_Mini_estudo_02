import csv
import math
import random
import time
import os

# parametros gerais
valores_d    = [1, 0.5, 0.1, 0.05]
TAXA_ENTRADA = 9
TAXA_SERVICO = 10
RECALC_CADA  = 50  # recalcula IC a cada N saídas
MIN_AMOSTRA = 150  # numero minimo para haver verificacao
CSV_RESUMO   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_resumo2.csv")

rho    = TAXA_ENTRADA / TAXA_SERVICO
Wq_teo = (rho / TAXA_SERVICO) / (1 - rho)

# forma do csv
with open(CSV_RESUMO, "w", newline="") as f:
    csv.writer(f).writerow([
        "d_alvo", "n_amostrado", "lambda", "mu", "rho",
        "media_estimada", "wq_teorico",
        "desvio_padrao", "ic_inferior", "ic_superior",
        "largura_ic", "teorico_no_ic", "duracao_wall_s"
    ])

print("─" * 60)
print("         SIMULAÇÃO M/M/1  –  λ=9  μ=10")
print(f"         Recalcula IC a cada {RECALC_CADA} saídas")
print("─" * 60)

for d_alvo in valores_d:

    # reinicio das variaveis para fazer a próxima simulação
    tempo_atual      = 0.0
    fila             = 0
    servidor_ocupado = False
    fila_chegada     = []

    proxima_chegada  = random.expovariate(TAXA_ENTRADA)
    proxima_saida    = float('inf')

    wf_n    = 0
    wf_mean = 0.0
    wf_M2   = 0.0

    margem      = float('inf')   # condição de parada: largura >= 2*d_alvo
    inicio_wall     = time.time()

    # simulação d_alvo é cada d estabelecido
    while margem>= 2*d_alvo:

        if proxima_chegada <= proxima_saida:
            # chegada
            tempo_atual = proxima_chegada

            if not servidor_ocupado:
                # atendido imediatamente
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


        # recalcula o ic a cada determinado numero de iteracoes
        if wf_n >= MIN_AMOSTRA and wf_n % RECALC_CADA == 0:
            desvio     = math.sqrt(wf_M2 / (wf_n - 1))
            margem     = 1.96 * desvio / math.sqrt(wf_n)
            largura_ic = 2 * margem


    # estatisticas 
    duracao_wall = time.time() - inicio_wall
    n      = wf_n
    media  = wf_mean
    desvio = math.sqrt(wf_M2 / (n - 1))
    margem = 1.96 * desvio / math.sqrt(n)
    ic_inf = media - margem
    ic_sup = media + margem
    no_ic  = ic_inf <= Wq_teo <= ic_sup

    # gravando no csv
    with open(CSV_RESUMO, "a", newline="") as f:
        csv.writer(f).writerow([
            d_alvo, n,
            TAXA_ENTRADA, TAXA_SERVICO, round(rho, 6),
            round(media,  8), round(Wq_teo, 8),
            round(desvio, 8), round(ic_inf, 8), round(ic_sup, 8),
            round(ic_sup - ic_inf, 8), no_ic, round(duracao_wall, 3)
        ])

    # impressao final no terminal
    print(f"\n{'='*60}")
    print(f"  d = {d_alvo}   →   parou com largura IC < {2*d_alvo}")
    print(f"{'='*60}")
    print(f"  Clientes amostrados          : {n:,}")
    print(f"  Média estimada  X̄(n)         : {media:.6f} s")
    print(f"  Valor teórico   E[Wq]        : {Wq_teo:.6f} s")
    print(f"  Desvio-padrão               : {desvio:.6f} s")
    print(f"  Largura do IC               : {ic_sup - ic_inf:.6f}")
    print(f"  Intervalo de confiança (95%) : [{ic_inf:.6f} , {ic_sup:.6f}]")
    print(f"  Teórico no IC               : {'✓ Sim' if no_ic else '✗ Não'}")
    print(f"  Wall-clock                  : {duracao_wall:.2f}s")

print("\n" + "─" * 60)
print(f"  CSV salvo em: {CSV_RESUMO}")
print("─" * 60)