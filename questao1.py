import csv
import math
import random
import time
import os

# parametros gerais
NUM_CLIENTES = [10**3, 10**5, 10**7, 10**9]
TAXA_ENTRADA = 9
TAXA_SERVICO = 10
# limite a partir do qual usa Welford (sem lista) para evitar explosão de memória e o processo ser morto
LIMITE_LISTA = 10**8
CSV_RESUMO   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_resumo.csv")

rho    = TAXA_ENTRADA / TAXA_SERVICO
Wq_teo = (rho / TAXA_SERVICO) / (1 - rho)

with open(CSV_RESUMO, "w", newline="") as f:
    csv.writer(f).writerow([
        "n_alvo", "n_amostrado", "lambda", "mu", "rho",
        "media_estimada", "wq_teorico",
        "desvio_padrao", "ic_inferior", "ic_superior",
        "teorico_no_ic", "duracao_wall_s"
    ])

print("─" * 60)
print("         SIMULAÇÃO M/M/1  –  λ=9  μ=10")
print("─" * 60)

for n_alvo in NUM_CLIENTES:

    # reinicio das variaveis
    tempo_atual      = 0.0
    fila             = 0
    servidor_ocupado = False
    fila_chegada     = []
    id_chegada       = 1
    id_saida         = 1
    proxima_chegada  = random.expovariate(TAXA_ENTRADA)
    proxima_saida    = float('inf')
    inicio_wall      = time.time()

    usar_welford = n_alvo > LIMITE_LISTA

    if usar_welford:
        # welford para evitar explodir a memória ram
        wf_n    = 0
        wf_mean = 0.0
        wf_M2   = 0.0
        print(f"\n  n = {n_alvo:,} → usando Welford (sem lista, evita OOM)")
    else:
        tempos_espera = []

    # loop geral
    while id_saida <= n_alvo:

        if proxima_chegada <= proxima_saida:
            tempo_atual = proxima_chegada
            if not servidor_ocupado:
                if usar_welford:
                    wf_n += 1
                    delta    = 0.0 - wf_mean
                    wf_mean += delta / wf_n
                    wf_M2   += delta * (0.0 - wf_mean)
                else:
                    tempos_espera.append(0.0)
                servidor_ocupado = True
                proxima_saida    = tempo_atual + random.expovariate(TAXA_SERVICO)
            else:
                fila += 1
                fila_chegada.append(tempo_atual)
            id_chegada     += 1
            proxima_chegada = tempo_atual + random.expovariate(TAXA_ENTRADA)

        else:
            tempo_atual = proxima_saida
            if fila > 0:
                fila -= 1
                t_chegada_proximo = fila_chegada.pop(0)
                wq = tempo_atual - t_chegada_proximo
                if usar_welford:
                    wf_n += 1
                    delta    = wq - wf_mean
                    wf_mean += delta / wf_n
                    wf_M2   += delta * (wq - wf_mean)
                else:
                    tempos_espera.append(wq)
                proxima_saida = tempo_atual + random.expovariate(TAXA_SERVICO)
            else:
                servidor_ocupado = False
                proxima_saida    = float('inf')
            id_saida += 1

    duracao_wall = time.time() - inicio_wall

    # estatisticas
    if usar_welford:
        n      = wf_n
        media  = wf_mean
        desvio = math.sqrt(wf_M2 / (n - 1))
    else:
        n         = len(tempos_espera)
        media     = sum(tempos_espera) / n
        soma_sq   = sum((x - media) ** 2 for x in tempos_espera)
        desvio    = math.sqrt(soma_sq / (n - 1))

    margem = 1.96 * desvio / math.sqrt(n)
    ic_inf = media - margem
    ic_sup = media + margem
    no_ic  = ic_inf <= Wq_teo <= ic_sup

    # gravando csv
    with open(CSV_RESUMO, "a", newline="") as f:
        csv.writer(f).writerow([
            n_alvo, n,
            TAXA_ENTRADA, TAXA_SERVICO, round(rho, 6),
            round(media,  8), round(Wq_teo, 8),
            round(desvio, 8), round(ic_inf, 8), round(ic_sup, 8),
            no_ic, round(duracao_wall, 3)
        ])

    # imprimir no terminal
    print(f"\n{'='*60}")
    print(f"  n = {n_alvo:,}   (wall-clock: {duracao_wall:.1f}s)")
    print(f"{'='*60}")
    print(f"  Clientes amostrados          : {n:,}")
    print(f"  Média estimada  X̄(n)         : {media:.6f} s")
    print(f"  Valor teórico   E[Wq]        : {Wq_teo:.6f} s")
    print(f"  Desvio-padrão               : {desvio:.6f} s")
    print(f"  Intervalo de confiança (95%) : [{ic_inf:.6f} , {ic_sup:.6f}]")
    print(f"  Teórico no IC               : {'✓ Sim' if no_ic else '✗ Não'}")

print("\n" + "─" * 60)
print(f"  CSV gerado: {CSV_RESUMO}")
print("─" * 60)