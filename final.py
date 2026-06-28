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
            "BIPEDv2/BIPEDv2/BIPED/edges/imgs/train/rgbr/real",
            "BIPEDv2/BIPEDv2/BIPED/edges/imgs/test/rgbr",
        ]
        origens_gabs = [
            "BIPEDv2/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real",
            "BIPEDv2/BIPEDv2/BIPED/edges/edge_maps/test/rgbr",
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
        Calcula q0 e todos os coeficientes do modelo fatorial 2³r usando a
        tabela de sinais, seguindo o método da teoria de planejamento fatorial.

        Procedimento correto para 2^k com replicação (r repetições):
          1. Agregar: calcular a média de cada combinação sobre as r repetições
             e todas as imagens (unidade experimental).
          2. Aplicar a tabela de sinais sobre as 8 médias → coeficientes q.
          3. Calcular SST = 2^k · r · n_imgs · Σ(q²)  (excluindo q0).
          4. Calcular a fração de variação explicada por cada termo: SS_termo / SST.
          5. Estimar o erro puro σ² como a média das variâncias intra-combinação.

        Codificação dos níveis (−1 / +1):
            A — Threshold      : 40 → −1  |  80 → +1
            B — Border type    : zero_padding → −1  |  replicate → +1
            C — Tamanho máscara: 3  → −1  |  7  → +1
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

        colunas_sinais = ["xA", "xB", "xC", "xAB", "xAC", "xBC", "xABC"]

        # ── Passo 1: média por combinação (agrega repetições e imagens) ───────
        # Cada combinação = (threshold, border_type, ksize)
        # A média é calculada sobre todas as repetições e imagens dessa combinação,
        # pois cada imagem é uma unidade experimental dentro da combinação.
        grupos = ["threshold", "border_type", "ksize", "xA", "xB", "xC",
                  "xAB", "xAC", "xBC", "xABC"]
        df_medias = df.groupby(grupos)[metricas].mean().reset_index()

        # Parâmetros do experimento
        k       = 3                        # número de fatores
        n_comb  = 2 ** k                   # 8 combinações
        r       = df["repeticao"].nunique() # número de repetições
        n_imgs  = df["nome_img"].nunique()  # número de imagens

        linhas = []
        for metrica in metricas:
            y_med = df_medias[metrica].values   # 8 médias (uma por combinação)

            linha = {"metrica": metrica, "n_combinacoes": n_comb, "r": r, "n_imgs": n_imgs}

            # ── Passo 2: coeficientes pela tabela de sinais ───────────────────
            # q0 = média geral das 8 médias de combinação
            q0 = y_med.mean()
            linha["q0"] = q0

            coefs = {}
            for col in colunas_sinais:
                sinais = df_medias[col].values
                coef   = (sinais * y_med).mean()   # média ponderada pelos sinais
                nome_q = "q" + col[1:]             # xA→qA, xAB→qAB, etc.
                linha[nome_q] = coef
                coefs[nome_q] = coef

            # ── Passo 3: SS de cada termo e SST ──────────────────────────────
            # SS_termo = 2^k · r · n_imgs · q²_termo
            # SST = soma de todos os SS_termo (excluindo q0)
            fator_ss = n_comb * r * n_imgs
            ss = {nome_q: fator_ss * (q ** 2) for nome_q, q in coefs.items()}
            sst = sum(ss.values())

            linha["SST"] = sst
            for nome_q, val in ss.items():
                linha[f"SS_{nome_q[1:]}"] = val   # SS_A, SS_B, etc.

            # ── Passo 4: fração de variação explicada ─────────────────────────
            for nome_q, val in ss.items():
                fracao = (val / sst * 100) if sst > 0 else 0.0
                linha[f"pct_{nome_q[1:]}"] = fracao   # pct_A, pct_B, etc.

            # ── Passo 5: erro experimental (variância pura entre repetições) ──
            var_por_comb = (
                df.groupby(["threshold", "border_type", "ksize"])[metrica]
                .var(ddof=1)
                .dropna()
            )
            sigma2 = var_por_comb.mean()
            sigma  = np.sqrt(sigma2) if sigma2 >= 0 else np.nan
            linha["erro_sigma2"] = sigma2
            linha["erro_sigma"]  = sigma

            linhas.append(linha)

        df_efeitos = pd.DataFrame(linhas)

        # ── Imprimir resultado no terminal ────────────────────────────────────
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",       220)
        pd.set_option("display.float_format", "{:.6f}".format)

        legenda = (
            "\nLegenda dos fatores:\n"
            "  A — Threshold       : nível baixo=40  | nível alto=80\n"
            "  B — Tratamento borda: nível baixo=zero_padding | nível alto=replicate\n"
            "  C — Tamanho máscara : nível baixo=3   | nível alto=7\n"
            "\nCoeficientes (tabela de sinais sobre médias por combinação):\n"
            "  q0              → média geral\n"
            "  qA, qB, qC      → efeitos principais\n"
            "  qAB, qAC, qBC   → interações de 2ª ordem\n"
            "  qABC            → interação de 3ª ordem\n"
            "\nVariação explicada:\n"
            "  SS_X   → soma de quadrados do termo X  (SS_X = 2^k · r · n_imgs · qX²)\n"
            "  SST    → soma total  (SST = Σ SS_X, excluindo q0)\n"
            "  pct_X  → porcentagem da variação total explicada pelo termo X\n"
            "\nErro:\n"
            "  erro_sigma2 → variância pura estimada entre repetições\n"
            "  erro_sigma  → desvio padrão do erro experimental\n"
        )

        # Tabela de coeficientes
        cols_coef = ["metrica", "q0", "qA", "qB", "qC", "qAB", "qAC", "qBC", "qABC"]
        # Tabela de variação explicada
        cols_pct  = ["metrica", "SST", "SS_A", "SS_B", "SS_C",
                     "SS_AB", "SS_AC", "SS_BC", "SS_ABC",
                     "pct_A", "pct_B", "pct_C",
                     "pct_AB", "pct_AC", "pct_BC", "pct_ABC",
                     "erro_sigma2", "erro_sigma"]

        print("\n" + "═" * 110)
        print("  ETAPA 8 — COEFICIENTES DO MODELO FATORIAL 2³r")
        print("═" * 110)
        print(legenda)
        print("── Coeficientes ──")
        print(df_efeitos[cols_coef].to_string(index=False))
        print("\n── Variação explicada (%) e erro experimental ──")
        print(df_efeitos[cols_pct].to_string(index=False))
        print("═" * 110 + "\n")

        # ── Salvar CSV ────────────────────────────────────────────────────────
        caminho_efeitos = caminho_csv.replace(".csv", "_efeitos.csv")
        df_efeitos.to_csv(caminho_efeitos, index=False, float_format="%.8f")
        print(f"  Coeficientes e frações salvas em: {caminho_efeitos}\n")

        return df_efeitos


# ══════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exp = Experimento()

    # Prepara o dataset automaticamente se a pasta ainda não tiver imagens
    if not any(PASTA_IMAGENS.glob("*.jpg")):
        print("Pasta de imagens vazia — preparando dataset...\n")
        exp.preparar_dataset()

    # Etapa 7 — execução do experimento (gera resultados_sobel.csv)
    exp.executar_experimento()

    # Etapa 8 — cálculo dos efeitos fatoriais (lê resultados_sobel.csv)
    exp.calcular_efeitos_fatoriais("resultados_sobel.csv")