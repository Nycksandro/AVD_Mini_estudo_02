"""
Experimento 2^k com r=5 repetições — Detecção de Borda com Sobel
=================================================================
Fatores (2 níveis cada):
  A - Threshold      : 40  | 80
  B - Tratamento borda: BORDER_CONSTANT (zero) | BORDER_REPLICATE
  C - Tamanho máscara: 3   | 7

Métricas coletadas: F1-Score, Precision, Recall, tempo de execução (ms), tempo de CPU (ms)
Requisitos da Etapa 7:
  - Fatores não analisados mantidos constantes
  - Ordem de combinações aleatorizada a cada repetição
  - Falhas/anomalias registradas com justificativa
  - Mesmo procedimento de medição em todas as combinações
"""

from typing import Callable, List
from itertools import product
from pathlib import Path
from datetime import datetime
import csv
import random
import shutil
import time
import traceback

import cv2
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DO EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

# Fatores e seus dois níveis
NIVEIS_THRESHOLD   = [40, 80]
NIVEIS_BORDER      = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]
NIVEIS_KSIZE       = [3, 7]
NOME_BORDER        = {cv2.BORDER_CONSTANT: "zero_padding", cv2.BORDER_REPLICATE: "replicate"}

# Fatores fixados (constantes)
DDEPTH             = cv2.CV_64F   # precisão numérica padrão para Sobel
NUM_REPETICOES     = 5            # r = 5
SEMENTE_ALEATORIA  = 42           # reprodutibilidade da aleatorização

# Caminhos
PASTA_IMAGENS    = Path("BIPEDv2_Adaptado/imagens")
PASTA_GABARITO   = Path("BIPEDv2_Adaptado/gabarito")
PASTA_DETECTADOS = Path("BIPEDv2_Adaptado/detectados")
ARQ_RESULTADOS   = "resultados_sobel.csv"
ARQ_ANOMALIAS    = "anomalias_sobel.csv"


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class Experimento:

    # ── Preparação do dataset ─────────────────────────────────────────────────

    def preparar_dataset(self) -> None:
        """
        Reorganiza o dataset BIPEDv2, unindo as partições de treino e teste
        em pastas únicas: imagens/, gabarito/ e detectados/.
        """

        def mover_arquivos(fontes: list[str], destino: Path) -> None:
            contador = 0
            for fonte in fontes:
                p = Path(fonte)
                if not p.exists():
                    print(f"  [AVISO] Pasta não encontrada: {fonte}")
                    continue
                for arq in p.glob("*.*"):
                    if arq.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        shutil.copy2(arq, destino / arq.name)
                        contador += 1
            print(f"  {contador} imagens copiadas para {destino}/")

        origens_imgs = [
            "BIPEDv2/BIPED/edges/imgs/train/rgbr/real",
            "BIPEDv2/BIPED/edges/imgs/test/rgbr",
        ]
        origens_gabs = [
            "BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real",
            "BIPEDv2/BIPED/edges/edge_maps/test/rgbr",
        ]

        for pasta in [PASTA_IMAGENS, PASTA_GABARITO, PASTA_DETECTADOS]:
            pasta.mkdir(parents=True, exist_ok=True)

        print("Preparando dataset...")
        mover_arquivos(origens_imgs, PASTA_IMAGENS)
        mover_arquivos(origens_gabs, PASTA_GABARITO)
        print("Dataset pronto.\n")

    # ── Detecção com Sobel ────────────────────────────────────────────────────

    def aplicar_sobel(
        self,
        img: cv2.typing.MatLike,
        ksize: int,
        border_type: int,
    ) -> cv2.typing.MatLike:
        """
        Aplica o filtro de Sobel e retorna a magnitude do gradiente como uint8.

        Fatores fixados:
          - ddepth    : CV_64F  (precisão padrão NumPy/OpenCV)
          - magnitude : L2 (cv2.magnitude das componentes X e Y)
          - biblioteca: OpenCV
        """
        grad_x = cv2.Sobel(img, DDEPTH, dx=1, dy=0, ksize=ksize, borderType=border_type)
        grad_y = cv2.Sobel(img, DDEPTH, dx=0, dy=1, ksize=ksize, borderType=border_type)

        magnitude = cv2.magnitude(grad_x.astype(np.float64), grad_y.astype(np.float64))
        resultado = cv2.convertScaleAbs(magnitude)
        return resultado

    # ── Threshold ─────────────────────────────────────────────────────────────

    @staticmethod
    def binarizar(img: cv2.typing.MatLike, threshold: int) -> cv2.typing.MatLike:
        """Binariza a imagem usando o threshold do fator A."""
        _, binaria = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binaria

    # ── Métricas ──────────────────────────────────────────────────────────────

    def calcular_metricas(
        self,
        predita: cv2.typing.MatLike,
        gabarito: cv2.typing.MatLike,
    ) -> List[float]:
        """
        Compara pixel a pixel (após binarização) e retorna [f1, precision, recall].
        Ambas as imagens já chegam binarizadas (0 ou 255).
        """
        borda_pred = predita  >= 127
        borda_real = gabarito >= 127

        vp = np.sum(borda_pred &  borda_real)
        fp = np.sum(borda_pred & ~borda_real)
        fn = np.sum(~borda_pred & borda_real)

        recall    = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return [f1, precision, recall]

    # ── Auxiliares de I/O ─────────────────────────────────────────────────────

    def _salvar_csv(self, registros: list[dict], caminho: str, label: str) -> None:
        if not registros:
            print(f"  [INFO] Nenhum registro de {label} para salvar.")
            return
        cabecalho = registros[0].keys()
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cabecalho)
            w.writeheader()
            w.writerows(registros)
        print(f"  {label} salvo em: {caminho}")

    # ── Experimento principal ─────────────────────────────────────────────────

    def executar_experimento(self) -> None:
        """
        Executa o experimento 2³ com r=5 repetições.

        Etapa 7 — boas práticas implementadas:
          • Ordem das combinações aleatorizada a cada repetição (seed fixa para
            reprodutibilidade, mas embaralhada por repetição).
          • Mesma rotina de medição para todas as combinações.
          • Falhas registradas separadamente com justificativa.
          • Fatores não analisados mantidos constantes (ddepth, biblioteca, SO,
            linguagem, ordem de imagens dentro de uma repetição).
        """

        # Gerar todas as 8 combinações de fatores
        combinacoes = list(product(NIVEIS_THRESHOLD, NIVEIS_BORDER, NIVEIS_KSIZE))

        arquivos = sorted(PASTA_IMAGENS.glob("*.jpg"))
        if not arquivos:
            raise FileNotFoundError(
                f"Nenhuma imagem .jpg encontrada em {PASTA_IMAGENS}. "
                "Execute preparar_dataset() primeiro."
            )

        resultados: list[dict] = []
        anomalias:  list[dict] = []

        rng = random.Random(SEMENTE_ALEATORIA)

        print(f"Iniciando experimento: {len(combinacoes)} combinações × "
              f"{len(arquivos)} imagens × {NUM_REPETICOES} repetições "
              f"= {len(combinacoes) * len(arquivos) * NUM_REPETICOES} execuções\n")

        for repeticao in range(1, NUM_REPETICOES + 1):
            # ── Aleatorizar a ordem das combinações nesta repetição ───────────
            ordem = combinacoes[:]
            rng.shuffle(ordem)

            print(f"── Repetição {repeticao}/{NUM_REPETICOES} "
                  f"(ordem das combinações aleatorizada) ──")

            for threshold, border_type, ksize in ordem:
                nome_borda  = NOME_BORDER[border_type]
                id_comb     = f"thr{threshold}_brd{nome_borda}_ks{ksize}"

                for caminho_img in arquivos:
                    # ── Carregar imagem ───────────────────────────────────────
                    img = cv2.imread(str(caminho_img), cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        anomalias.append({
                            "repeticao":    repeticao,
                            "combinacao":   id_comb,
                            "imagem":       caminho_img.name,
                            "tipo":         "leitura_img",
                            "descricao":    "cv2.imread retornou None — arquivo corrompido ou formato inválido.",
                            "acao":         "Medição descartada.",
                            "horario":      datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        })
                        continue

                    # ── Carregar gabarito ─────────────────────────────────────
                    caminho_gab = (
                        PASTA_GABARITO / caminho_img.with_suffix(".png").name
                    )
                    img_gab = cv2.imread(str(caminho_gab), cv2.IMREAD_GRAYSCALE)

                    if img_gab is None:
                        anomalias.append({
                            "repeticao":    repeticao,
                            "combinacao":   id_comb,
                            "imagem":       caminho_img.name,
                            "tipo":         "leitura_gabarito",
                            "descricao":    f"Gabarito não encontrado: {caminho_gab}",
                            "acao":         "Métricas não calculadas; execução registrada sem F1.",
                            "horario":      datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        })
                        img_gab = None   # será tratado abaixo

                    # ── Medição (mesmo procedimento para todas as combinações) ─
                    horario_inicio   = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    t_wall_inicio    = time.perf_counter()
                    t_cpu_inicio     = time.process_time()

                    try:
                        resultado_borda = self.aplicar_sobel(img, ksize, border_type)
                        resultado_bin   = self.binarizar(resultado_borda, threshold)
                    except Exception as exc:
                        anomalias.append({
                            "repeticao":    repeticao,
                            "combinacao":   id_comb,
                            "imagem":       caminho_img.name,
                            "tipo":         "erro_deteccao",
                            "descricao":    str(exc),
                            "acao":         "Medição descartada. Traceback: " + traceback.format_exc(limit=2),
                            "horario":      datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        })
                        continue

                    t_wall_fim  = time.perf_counter()
                    t_cpu_fim   = time.process_time()
                    horario_fim = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                    tempo_ms    = (t_wall_fim - t_wall_inicio) * 1000
                    cpu_time_ms = (t_cpu_fim  - t_cpu_inicio)  * 1000

                    # ── Métricas ──────────────────────────────────────────────
                    if img_gab is not None:
                        img_gab_bin = self.binarizar(img_gab, threshold)
                        f1, prec, rec = self.calcular_metricas(resultado_bin, img_gab_bin)
                    else:
                        f1 = prec = rec = None   # gabarito ausente

                    # ── Salvar imagem detectada ───────────────────────────────
                    nome_saida = (
                        f"{caminho_img.stem}_{id_comb}.jpg"
                    )
                    cv2.imwrite(str(PASTA_DETECTADOS / nome_saida), resultado_bin)

                    # ── Registrar resultado ───────────────────────────────────
                    resultados.append({
                        "repeticao":      repeticao,
                        "nome_img":       caminho_img.name,
                        "threshold":      threshold,
                        "border_type":    nome_borda,
                        "ksize":          ksize,
                        "combinacao":     id_comb,
                        "f1score":        f1,
                        "precision":      prec,
                        "recall":         rec,
                        "tempo_ms":       round(tempo_ms,    4),
                        "cpu_time_ms":    round(cpu_time_ms, 4),
                        "horario_inicio": horario_inicio,
                        "horario_fim":    horario_fim,
                    })

            print(f"   Repetição {repeticao} concluída — "
                  f"{len(resultados)} registros acumulados.\n")

        # ── Persistir resultados e anomalias ──────────────────────────────────
        self._salvar_csv(resultados, ARQ_RESULTADOS, "Resultados")
        self._salvar_csv(anomalias,  ARQ_ANOMALIAS,  "Anomalias")

        self.gerar_relatorio(ARQ_RESULTADOS)

    # ── Relatório agregado ────────────────────────────────────────────────────

    def gerar_relatorio(self, caminho_csv: str) -> None:
        """
        Lê o CSV de resultados e imprime um relatório agregado por combinação,
        com média e desvio padrão de F1, tempo de execução e tempo de CPU.
        """
        df = pd.read_csv(caminho_csv)

        # Remover linhas sem F1 (gabarito ausente)
        df_valido = df.dropna(subset=["f1score"])
        descartados = len(df) - len(df_valido)
        if descartados:
            print(f"  [INFO] {descartados} linha(s) sem gabarito excluídas do relatório.")

        relatorio = (
            df_valido
            .groupby(["threshold", "border_type", "ksize", "combinacao"])
            .agg(
                f1_medio    =("f1score",     "mean"),
                f1_std      =("f1score",     "std"),
                prec_media  =("precision",   "mean"),
                rec_media   =("recall",      "mean"),
                tempo_medio =("tempo_ms",    "mean"),
                tempo_std   =("tempo_ms",    "std"),
                cpu_medio   =("cpu_time_ms", "mean"),
                cpu_std     =("cpu_time_ms", "std"),
                n_obs       =("f1score",     "count"),
            )
            .reset_index()
            .sort_values("f1_medio", ascending=False)
        )

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",       200)
        pd.set_option("display.float_format", "{:.4f}".format)

        print("\n" + "═" * 80)
        print("  RELATÓRIO FINAL — médias por combinação de fatores")
        print("═" * 80)
        print(relatorio.to_string(index=False))
        print("═" * 80 + "\n")

        # Salvar relatório agregado
        caminho_relatorio = caminho_csv.replace(".csv", "_relatorio.csv")
        relatorio.to_csv(caminho_relatorio, index=False, float_format="%.6f")
        print(f"  Relatório agregado salvo em: {caminho_relatorio}\n")

    # ── Etapa 8: Cálculo dos efeitos fatoriais ────────────────────────────────

    def calcular_efeitos_fatoriais(
        self,
        caminho_csv: str = "resultados_sobel.csv",
        metricas: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calcula q0 e todos os coeficientes do modelo fatorial 2³ usando a
        tabela de sinais, para cada métrica solicitada.

        Modelo:
            y = q0 + qA·xA + qB·xB + qC·xC
                   + qAB·xA·xB + qAC·xA·xC + qBC·xB·xC
                   + qABC·xA·xB·xC + ε

        Codificação dos níveis (−1 / +1):
            A  — Threshold      : 40 → −1  |  80 → +1
            B  — Border type    : BORDER_CONSTANT (zero_padding) → −1  |  BORDER_REPLICATE → +1
            C  — Tamanho máscara: 3  → −1  |  7  → +1

        Cada coeficiente q é calculado como:
            q_termo = (1 / N) · Σ (sinal_termo_i · y_i)
        onde N = número total de observações (8 combinações × r repetições × n_imgs).

        Interpretação dos resultados impressos:
            • Efeito alto  → o fator (ou interação) altera fortemente a métrica.
            • Efeito baixo → o fator atua de forma mais independente / pouco influente.
            • Erro experimental (σ²) → variabilidade residual entre repetições.
        """
        if metricas is None:
            metricas = ["f1score", "precision", "recall", "tempo_ms", "cpu_time_ms"]

        df = pd.read_csv(caminho_csv).dropna(subset=metricas)

        # ── Codificar fatores em −1 / +1 ─────────────────────────────────────
        mapa_threshold = {40: -1, 80: +1}
        mapa_border    = {"zero_padding": -1, "replicate": +1}
        mapa_ksize     = {3: -1, 7: +1}

        df["xA"] = df["threshold"].map(mapa_threshold)
        df["xB"] = df["border_type"].map(mapa_border)
        df["xC"] = df["ksize"].map(mapa_ksize)

        # Verificar se todos os valores foram mapeados corretamente
        for col, nome in [("xA", "threshold"), ("xB", "border_type"), ("xC", "ksize")]:
            if df[col].isna().any():
                raise ValueError(
                    f"Valores inesperados na coluna '{nome}'. "
                    f"Valores únicos encontrados: {df[nome].unique()}"
                )

        # ── Colunas de interação ──────────────────────────────────────────────
        df["xAB"]  = df["xA"] * df["xB"]
        df["xAC"]  = df["xA"] * df["xC"]
        df["xBC"]  = df["xB"] * df["xC"]
        df["xABC"] = df["xA"] * df["xB"] * df["xC"]

        termos = {
            "q0"   : None,       # intercepto: média geral
            "qA"   : "xA",
            "qB"   : "xB",
            "qC"   : "xC",
            "qAB"  : "xAB",
            "qAC"  : "xAC",
            "qBC"  : "xBC",
            "qABC" : "xABC",
        }

        N = len(df)

        # ── Calcular coeficientes para cada métrica ───────────────────────────
        linhas = []
        for metrica in metricas:
            y = df[metrica].values
            linha = {"metrica": metrica, "N_obs": N}

            for nome_termo, col_sinal in termos.items():
                if col_sinal is None:          # q0 = média geral
                    coef = y.mean()
                else:
                    sinais = df[col_sinal].values
                    coef   = (sinais * y).mean()
                linha[nome_termo] = coef

            # ── Erro experimental: variância entre repetições por combinação ──
            # Para cada combinação, calculamos a variância entre as r repetições
            # e depois tiramos a média — isso é a estimativa do erro puro σ².
            var_por_comb = (
                df.groupby(["threshold", "border_type", "ksize"])[metrica]
                .var(ddof=1)
                .dropna()
            )
            sigma2 = var_por_comb.mean()       # média das variâncias (erro puro)
            sigma  = np.sqrt(sigma2) if sigma2 >= 0 else np.nan

            linha["erro_sigma2"] = sigma2
            linha["erro_sigma"]  = sigma

            linhas.append(linha)

        df_efeitos = pd.DataFrame(linhas)

        # ── Imprimir tabela no terminal ───────────────────────────────────────
        colunas_ordem = [
            "metrica", "N_obs",
            "q0", "qA", "qB", "qC",
            "qAB", "qAC", "qBC", "qABC",
            "erro_sigma2", "erro_sigma",
        ]
        df_efeitos = df_efeitos[colunas_ordem]

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",       200)
        pd.set_option("display.float_format", "{:.6f}".format)

        legenda = (
            "\nLegenda dos fatores:\n"
            "  A — Threshold       : nível baixo=40  | nível alto=80\n"
            "  B — Tratamento borda: nível baixo=zero_padding | nível alto=replicate\n"
            "  C — Tamanho máscara : nível baixo=3   | nível alto=7\n"
            "\nInterpretação:\n"
            "  q0           → média geral das respostas\n"
            "  qA, qB, qC   → efeito principal de cada fator\n"
            "  qAB,qAC,qBC  → interação de 2ª ordem (efeito de um depende do outro)\n"
            "  qABC         → interação de 3ª ordem\n"
            "  erro_sigma2  → variância do erro experimental (entre repetições)\n"
            "  erro_sigma   → desvio padrão do erro experimental\n"
        )

        print("\n" + "═" * 100)
        print("  ETAPA 8 — COEFICIENTES DO MODELO FATORIAL 2³")
        print("═" * 100)
        print(legenda)
        print(df_efeitos.to_string(index=False))
        print("═" * 100 + "\n")

        # ── Salvar CSV ────────────────────────────────────────────────────────
        caminho_efeitos = caminho_csv.replace(".csv", "_efeitos.csv")
        df_efeitos.to_csv(caminho_efeitos, index=False, float_format="%.8f")
        print(f"  Coeficientes salvos em: {caminho_efeitos}\n")

        return df_efeitos


# ══════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exp = Experimento()

    # Descomente a linha abaixo na primeira execução para reorganizar o dataset:
    # exp.preparar_dataset()

    # Etapa 7 — execução do experimento (gera resultados_sobel.csv)
    exp.executar_experimento()

    # Etapa 8 — cálculo dos efeitos fatoriais (lê resultados_sobel.csv)
    exp.calcular_efeitos_fatoriais("resultados_sobel.csv")