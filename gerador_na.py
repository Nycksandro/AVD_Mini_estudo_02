import numpy as np
import math

class gerador_na:
    def __init__(self, seed=67, a=214013, c=253101, m=2**32): # Incializador seta os paremetros
        """
        ### Gerador Linear Congruencial 
        
        * Parâmetros
            * seed : `int`
                Valor da semente (e inicial) do LCG.
            * a : `int`
                Multiplicador.
            * c : `int`
                Incrementador.
            * m : `int`
                Módulo.
        """
        self.seed = seed
        self.a = 214013
        self.c = 253101
        self.m = 2**32 

    def lcg(self): #Fonte: https://www-quantstart-com.translate.goog/articles/linear-congruential-generators-in-python/?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=sge
        """
        Essa função aplica a fórmula matemática do LCG:
            X_n = (a * X_{n-1} + c) % m
        
        Utiliza os paramêtros setados da inicializador da classe.

        * Retorno: Número entre 0-1: `float`
        """
        # Aplica a fórmula matemática do LCG: X_n = (a * X_{n-1} + c) % m
        self.seed = (self.a * self.seed + self.c) % self.m
        
        # Normaliza para retornar um float entre 0.0 e 1.0
        return self.seed / self.m
    
    def tempo_proxima_chegada(self, taxa_entrada): # Definido com base no final do slide
        u = self.lcg()
        return -math.log(1.0 - u) / taxa_entrada

    def tempo_atendimento(self, taxa_servico): # Definido com base no final do slide
        u = self.lcg()
        return -math.log(1.0 - u) / taxa_servico