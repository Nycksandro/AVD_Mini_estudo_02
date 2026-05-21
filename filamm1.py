from gerador_na import gerador_na
import math
import time

gerador_num_aleatorio = gerador_na(seed=67) # Instanciando a classe do gerador de núemros aleatórios

#Parametros
NUM_CLIENTES = 10**6 # clientes
TAXA_ENTRADA = 9 # por segundo
TAXA_SERVICO = 10 # por segundo

tempo_atual = 0
fila = 0 
servidor_ocupado = False

tempos_espera = []
historico_tempo_chegada = {} 
id_cliente_chegada = 1
id_cliente_saida = 1

proxima_chegada = tempo_atual + gerador_num_aleatorio.tempo_proxima_chegada(TAXA_ENTRADA)
proxima_saida = float('inf') 

print(f"--- INÍCIO DA SIMULAÇÃO (M/M/1) ---")

while id_cliente_saida <= NUM_CLIENTES: # Processa o último cliente ser atendido
    
    if proxima_chegada < proxima_saida:
        # Chegada
        tempo_atual = proxima_chegada
        historico_tempo_chegada[id_cliente_chegada] = tempo_atual
        
        print(f"[{tempo_atual:7.2f}] EVENTO: Cliente {id_cliente_chegada} CHEGOU.")
        
        if not servidor_ocupado:
            servidor_ocupado = True
            # Tempo de serviço exponencial
            duracao = gerador_num_aleatorio.tempo_atendimento(TAXA_SERVICO)
            proxima_saida = tempo_atual + duracao
            print(f"          -> Servidor livre. Cliente {id_cliente_chegada} começou a ser atendido (Duração: {duracao:.2f}).")
        else:
            fila += 1
            print(f"          -> Servidor ocupado. Cliente {id_cliente_chegada} foi para a fila (Tamanho da fila: {fila}).")
        
        id_cliente_chegada += 1
        # Próxima chegada exponencial
        proxima_chegada = tempo_atual + gerador_num_aleatorio.tempo_proxima_chegada(TAXA_ENTRADA)
        
    else:
        # Fim do atendimento
        tempo_atual = proxima_saida
        print(f"[{tempo_atual:7.2f}] EVENTO: Cliente {id_cliente_saida} FOI EMBORA.")
        
        if fila > 0:
            fila -= 1
            id_proximo_atendido = id_cliente_saida + 1
            tempo_espera_fila = tempo_atual - historico_tempo_chegada[id_proximo_atendido]
            tempos_espera.append(tempo_espera_fila)
            
            # Tempo para o próximo cliente
            duracao = gerador_num_aleatorio.tempo_atendimento(TAXA_SERVICO)
            proxima_saida = tempo_atual + duracao
            print(f"          -> Próximo da fila (Cliente {id_proximo_atendido}) assume o servidor (Espera na fila: {tempo_espera_fila:.2f}).")
        else:
            servidor_ocupado = False
            proxima_saida = float('inf')
            print(f"          -> Fila vazia. Servidor agora está OCIOSO.")
            
        id_cliente_saida += 1

print("\n--- RESULTADOS DA SIMULAÇÃO ---")
while len(tempos_espera) < NUM_CLIENTES:
    tempos_espera.append(0.0)

#Estimando tempo médio de espera

n = len(tempos_espera)
media_estimada = sum(tempos_espera) / n

# Calculo da Variância Amostral (S^2)
soma_quadrados_desvios = sum((x - media_estimada) ** 2 for x in tempos_espera)
variancia_amostral = soma_quadrados_desvios / (n - 1)
desvio_padrao = math.sqrt(variancia_amostral)

# Calculo do Erro Padrão e a Margem de Erro (Z = 1.96 para 95%)
Z = 1.96
erro_padrao = desvio_padrao / math.sqrt(n)
margem_erro = Z * erro_padrao

ic_inferior = media_estimada - margem_erro
ic_superior = media_estimada + margem_erro

# Calculo do valor esperado téorico E[X]
rho = TAXA_ENTRADA / TAXA_SERVICO
Wq_teorico = (rho * (1 / TAXA_SERVICO)) / (1 - rho)

# Resultados 
print("\n" + "="*50)
print("             ANÁLISE ESTATÍSTICA M/M/1")
print("="*50)
print(f"Número de clientes simulados (n) : {n}")
print(f"Taxa de Chegada (lambda)         : {TAXA_ENTRADA} clientes/s")
print(f"Taxa de Atendimento (mu)         : {TAXA_SERVICO} clientes/s")
print("-"*50)
print(f"Tempo Médio Estimado X(n)        : {media_estimada:.5f} segundos")
print(f"Valor Teórico Esperado E[X]      : {Wq_teorico:.5f} segundos")
print("-"*50)
print(f"Intervalo de Confiança (95%)     : [{ic_inferior:.5f} , {ic_superior:.5f}]")

# Validação do Intervalo
if ic_inferior <= Wq_teorico <= ic_superior:
    print("\nValor teórico está dentro do intervalo de confiança")
else:
    print("\nValor teórico ficou fora do intervalo de confiança")
print("="*50)