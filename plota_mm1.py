import csv
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_RESUMO = "resultados_resumo.csv"

# ── Lê o CSV ──────────────────────────────────────────────────────────────────
ns, medias, ic_infs, ic_sups, wq_teos = [], [], [], [], []

with open(CSV_RESUMO, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ns.append(int(row["n_alvo"]))
        medias.append(float(row["media_estimada"]))
        ic_infs.append(float(row["ic_inferior"]))
        ic_sups.append(float(row["ic_superior"]))
        wq_teos.append(float(row["wq_teorico"]))

erros_inf = [m - i for m, i in zip(medias, ic_infs)]
erros_sup = [s - m for m, s in zip(medias, ic_sups)]
Wq_teo = wq_teos[0]  # constante para todas as rodadas

# ── Estilo ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#c8ccd8",
    "xtick.color":      "#6b7080",
    "ytick.color":      "#6b7080",
    "grid.color":       "#1e2130",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "text.color":       "#c8ccd8",
    "font.family":      "monospace",
})

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("#0f1117")

# Banda do IC
ax.fill_between(ns, ic_infs, ic_sups,
                color="#3b82f6", alpha=0.15, label="IC 95%")

# Linha da média estimada
ax.plot(ns, medias,
        color="#3b82f6", linewidth=2, marker="o", markersize=7,
        markerfacecolor="#0f1117", markeredgewidth=2,
        label="Média estimada $\\bar{X}(n)$", zorder=3)

# Barras de erro explícitas
ax.errorbar(ns, medias, yerr=[erros_inf, erros_sup],
            fmt="none", ecolor="#3b82f6", elinewidth=1.2,
            capsize=6, capthick=1.5, zorder=4)

# Linha do valor teórico
ax.axhline(Wq_teo,
           color="#f97316", linewidth=1.8, linestyle="--",
           label=f"$E[W_q]$ teórico = {Wq_teo:.4f} s", zorder=2)

# Anotações nos pontos
for x, y, yi, ys in zip(ns, medias, ic_infs, ic_sups):
    label = f"{y:.4f}\n[{yi:.4f}, {ys:.4f}]"
    ax.annotate(label,
                xy=(x, y), xytext=(0, 18),
                textcoords="offset points",
                ha="center", fontsize=7.5,
                color="#94a3b8",
                arrowprops=dict(arrowstyle="-", color="#2a2d3a", lw=0.8))

# Eixos
ax.set_xscale("log")
ax.set_xlabel("n  (clientes simulados — escala log)", fontsize=11, labelpad=10)
ax.set_ylabel("Tempo médio de espera na fila  (s)", fontsize=11, labelpad=10)
ax.set_title("Convergência da Simulação M/M/1\n"
             r"$\lambda=9$ cl/s    $\mu=10$ cl/s    $\rho=0{,}9$",
             fontsize=13, pad=16, color="#e2e8f0")

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"$10^{{{int(math.log10(v))}}}$" if v > 0 else ""))
ax.grid(True, which="both")
ax.tick_params(labelsize=9)

legend = ax.legend(fontsize=9.5, framealpha=0.15,
                   facecolor="#1e2130", edgecolor="#2a2d3a",
                   labelcolor="#c8ccd8", loc="upper right")

plt.tight_layout(pad=1.6)
plt.savefig("grafico_mm1.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Gráfico salvo em: grafico_mm1.png")
plt.show()