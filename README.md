# NNDL Saliency Project — Team [A/B/C]

Progetto di visual saliency prediction (fixation prediction) su SALICON.
Vedi `piano_sviluppo_2settimane.pdf` (fuori da questo repo, v2) per il piano completo.

**Disegno sperimentale (v2, vedi `configs/experiments.yaml`):** B0 (center prior) →
B1 (ResNet18+decoder, MSE) → M1 (+ skip connections, STESSA loss di B1) → M1-L (stessa
M1, loss → CC+KLD) → G (Adaptive Center Prior, **obbligatorio**) → M2 (encoder Transformer
leggero). Ogni passaggio cambia una sola cosa alla volta rispetto al precedente — non
mischiare mai cambio di architettura e cambio di loss nello stesso esperimento.

## Struttura

```
configs/        # data.yaml (split, path dataset) + experiments.yaml (disegno sperimentale B0..M2)
src/data/       # Dataset, DataLoader, creazione split, audit
src/models/     # B0, B1 (baseline), M1/M1-L (skip connections), G (adaptive center prior), M2 (encoder Transformer)
src/losses/     # MSE, CC-loss, KLD
scripts/        # entrypoint: download, audit, train, evaluate, bootstrap CI
tests/          # unit test (metriche, shape, split leakage)
results/        # manifest.csv delle run, NON committare file pesanti
notebooks/      # notebook Colab "bootstrap" per i run lunghi (incl. copia dataset su disco locale VM)
```

## Giorno 1 — cosa fare, in ordine

1. **Un solo membro** crea il repo GitHub e invita gli altri due come collaboratori. Ognuno clona in locale.
2. Ognuno crea l'ambiente Python:
   - **Su M1 Max (sviluppo/debug):** `conda env create -f environment-mac.yml` poi `conda activate nndl-saliency`
   - **Su Colab (run pesanti):** apri `notebooks/colab_bootstrap.ipynb`, esegue da solo l'installazione da `requirements-colab.txt`
3. **Persona A**: segue `scripts/download_salicon.py` per scaricare SALICON da Kaggle (serve un account Kaggle + API token, vedi sotto).
4. **Persona A**: appena scaricato, lancia `scripts/audit_dataset.py` — questo risponde SUBITO alla domanda critica: *ci sono le fixation coordinates o solo le density map?* Da questo dipende se potrete usare NSS/sAUC oltre a CC/SIM/KLD.
5. Tutti: verificare che l'ambiente locale (M1) faccia un forward pass di prova (vedi `scripts/smoke_test.py`, arriva al giorno 2 insieme al dataloader).

## Setup Kaggle API (serve al passo 3)

1. Vai su https://www.kaggle.com/settings → "Create New Token" → scarica `kaggle.json`.
2. Metti il file in `~/.kaggle/kaggle.json` (Mac/Linux) e fai `chmod 600 ~/.kaggle/kaggle.json`.
3. Su Colab: carica `kaggle.json` con lo snippet incluso in `notebooks/colab_bootstrap.ipynb`.

## Convenzioni

- Nessun path assoluto locale nel codice: tutti i path passano da `configs/data.yaml`.
- Split (10k train / 2.5k tuning / 2.5k test interno, seed fisso) e manifest vanno versionati in `results/` (file leggeri, es. CSV con ID) — non i dati o i checkpoint.
- Ogni esperimento (B0/B1/M1/M1-L/G/M2) legge la propria configurazione da `configs/experiments.yaml`, niente iperparametri hardcoded negli script.
- **Su Colab: copiare sempre il dataset dal Drive montato al disco locale della VM prima di un training** (vedi `notebooks/colab_bootstrap.ipynb`, cella 6bis) — leggere da Drive montato è il vero collo di bottiglia, non il calcolo.
- Se il tempo stringe, l'ordine di taglio è: MIT1003 → M2 ridotto. **Mai** B0/B1/M1/M1-L/G.
