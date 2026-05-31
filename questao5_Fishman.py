# Utilizamos o código base da questão 01 e adaptamos pra essa questão

import csv
import math
import random
import time
import pandas as pd
import os

# parametros gerais
NUM_CLIENTES_VALIDOS = 10**3
TAXA_ENTRADA = 9.5
TAXA_SERVICO = 10
NUM_REPETICAO = 30

# Parâmetro de Fishman: Quantas vezes a série deve cruzar a média acumulada
ALVO_CRUZAMENTOS_FISHMAN = 25

CSV_RESUMO   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_resumo5_Fishman.csv")

rho    = TAXA_ENTRADA / TAXA_SERVICO
Wq_teo = (rho / TAXA_SERVICO) / (1 - rho)

with open(CSV_RESUMO, "w", newline="") as f:
    csv.writer(f).writerow([
        "n_iteracao", "n_alvo", "n_amostrado", 
        "lambda", "mu", "rho",
        "media_estimada", "wq_teorico", "vies",
        "desvio_padrao", "ic_inferior", "ic_superior",
        "teorico_no_ic", "duracao_wall_s"
    ])

print("─" * 60)
print("         SIMULAÇÃO M/M/1  –  λ=9.5  μ=10")
print("─" * 60)

for iteracao in range(1, NUM_REPETICAO+1): # Pra executar 30 vezes
    print(f"\nIteração: {iteracao}")

# reinicio das variaveis de estado da fila
    tempo_atual      = 0.0
    fila             = 0
    servidor_ocupado = False
    fila_chegada     = []
    
    proxima_chegada  = random.expovariate(TAXA_ENTRADA)
    proxima_saida    = float('inf')
    inicio_wall      = time.time()

    # Controles da Heurística de Fishman
    transiente_eliminado = False
    clientes_descartados = 0
    
    soma_transiente = 0.0
    qtd_transiente = 0
    wq_anterior = None
    contador_cruzamentos = 0

    # Armazenamento pós-transiente
    tempos_espera = []
    id_saida_valida = 0

    # O loop roda até coletarmos a quantidade alvo de clientes VÁLIDOS
    while id_saida_valida < NUM_CLIENTES_VALIDOS:

        if proxima_chegada <= proxima_saida:
            tempo_atual = proxima_chegada
            
            if not servidor_ocupado:
                wq = 0.0
                
                # Avaliação de Fishman para o cliente atual
                if not transiente_eliminado:
                    qtd_transiente += 1
                    soma_transiente += wq
                    media_acumulada = soma_transiente / qtd_transiente
                    
                    if wq_anterior is not None:
                        # Verifica se cruzou a média: um estava acima/igual e o outro abaixo (ou vice-versa)
                        if (wq_anterior >= media_acumulada and wq < media_acumulada) or \
                           (wq_anterior <= media_acumulada and wq > media_acumulada):
                            contador_cruzamentos += 1
                    
                    wq_anterior = wq
                    
                    if contador_cruzamentos >= ALVO_CRUZAMENTOS_FISHMAN:
                        transiente_eliminado = True
                    else:
                        clientes_descartados += 1
                
                # Se o transiente já acabou
                if transiente_eliminado:
                    tempos_espera.append(wq)
                    id_saida_valida += 1

                servidor_ocupado = True
                proxima_saida    = tempo_atual + random.expovariate(TAXA_SERVICO)
            else:
                fila += 1
                fila_chegada.append(tempo_atual)
            
            proxima_chegada = tempo_atual + random.expovariate(TAXA_ENTRADA)

        else:
            tempo_atual = proxima_saida
            
            if fila > 0:
                fila -= 1
                t_chegada_proximo = fila_chegada.pop(0)
                wq = tempo_atual - t_chegada_proximo
                
                # Avaliação de Fishman para o cliente atual
                if not transiente_eliminado:
                    qtd_transiente += 1
                    soma_transiente += wq
                    media_acumulada = soma_transiente / qtd_transiente
                    
                    if wq_anterior is not None:
                        # Verifica se cruzou a média
                        if (wq_anterior >= media_acumulada and wq < media_acumulada) or \
                           (wq_anterior <= media_acumulada and wq > media_acumulada):
                            contador_cruzamentos += 1
                    
                    wq_anterior = wq
                    
                    if contador_cruzamentos >= ALVO_CRUZAMENTOS_FISHMAN:
                        transiente_eliminado = True
                    else:
                        clientes_descartados += 1
                
                # Se o transiente já acabou
                if transiente_eliminado:
                    tempos_espera.append(wq)
                    id_saida_valida += 1
                
                proxima_saida = tempo_atual + random.expovariate(TAXA_SERVICO)
            else:
                servidor_ocupado = False
                proxima_saida    = float('inf')

    duracao_wall = time.time() - inicio_wall

    n         = len(tempos_espera)
    media     = sum(tempos_espera) / n
    soma_sq   = sum((x - media) ** 2 for x in tempos_espera)
    desvio    = math.sqrt(soma_sq / (n - 1))

    margem = 1.96 * desvio / math.sqrt(n)
    ic_inf = media - margem
    ic_sup = media + margem
    no_ic  = ic_inf <= Wq_teo <= ic_sup
    vies = media - Wq_teo # Calculando o viés

    # gravando csv
    with open(CSV_RESUMO, "a", newline="") as f:
        csv.writer(f).writerow([
            iteracao, 
            NUM_CLIENTES_VALIDOS, 
            n,
            TAXA_ENTRADA, 
            TAXA_SERVICO, 
            round(rho, 6),
            round(media, 8), 
            round(Wq_teo, 8), 
            round(vies, 8),
            round(desvio, 8), 
            round(ic_inf, 8), 
            round(ic_sup, 8),
            no_ic, 
            round(duracao_wall, 3)
        ])

    # imprimir no terminal
    print(f"\n{'='*60}")
    print(f"  n_valido = {NUM_CLIENTES_VALIDOS:,} | Descartados no transiente: {clientes_descartados}")
    print(f"{'='*60}")
    print(f"  Média estimada  X̄(n)         : {media:.6f} s")
    print(f"  Valor teórico   E[Wq]        : {Wq_teo:.6f} s")
    print(f"  Viés : {vies}")
    print(f"  Desvio-padrão               : {desvio:.6f} s")
    print(f"  Intervalo de confiança (95%) : [{ic_inf:.6f} , {ic_sup:.6f}]")
    print(f"  Teórico no IC               : {'✓ Sim' if no_ic else '✗ Não'}")
    
# Calculando os valores médios de cada coluna
df = pd.read_csv(CSV_RESUMO)
linha = ["Média Global"]
for coluna in df.columns[1:]:
    media = df[coluna].mean() # Criando uma linha
    linha.append(media)

linha_final_csv = pd.DataFrame([linha], columns=df.columns)

linha_final_csv.to_csv(CSV_RESUMO, mode='a', index=False, header=False, encoding='utf-8') # Colocando a média no final do csv

# Imprimir no terminal os valores da média global
print(f"\n{'='*60}")
print(f"  {linha[0]}   ({NUM_REPETICAO} Repetições)")
print(f"{'='*60}")
print(f"  Média estimada  X̄(n)         : {linha[6]:.6f} s")
print(f"  Valor teórico   E[Wq]        : {linha[7]:.6f} s")
print(f"  Viés : {linha[8]}")
print("\n" + "─" * 60)
print(f"  CSV gerado: {CSV_RESUMO}")
print("─" * 60)