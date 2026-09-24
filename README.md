# NNDL Saliency Project

Predizione di mappe di salienza su SALICON. Il progetto confronta i modelli cambiando una componente alla volta, con split e protocollo di valutazione condivisi.

## Stato del progetto

| Blocco | Stato al 24/09/2026 |
| --- | --- |
| Dataset, split e preprocessing | Implementati; manifest fisso in `results/split_manifest.csv` |
| Metriche e loss | CC/SIM/KLD e loss implementate e testate. Fix del 24/09: `kld()` in `src/metrics.py` applicava l'epsilon due volte (dentro la normalizzazione e di nuovo nel rapporto logaritmico); rimossa la doppia applicazione, default allineato a `1e-6` (stesso valore di `data.yaml`, già usato in produzione). NSS/sAUC implementate e testate, mantenute nel codice ma non ancora nella tabella principale del report (decisione presa il 24/09) |
| B0 (center prior) | Risultato ufficiale prodotto su `tuning`: CC 0.538, SIM 0.570, KLD 0.627. Non richiede re-training (nessun backprop) |
| B1 (ResNet18 + decoder, MSE) | Risultato su protocollo pre-freeze (3 epoche, train completo): CC 0.867, SIM 0.765, KLD 0.208. **Da rieseguire con il protocollo congelato (epochs=30, cartella pulita)** |
| M1 (multi-scala, MSE) | Risultato pre-freeze: CC 0.873, SIM 0.769, KLD 0.203. Significativo vs B1 (bootstrap CI 95%, delta CC/SIM/KLD tutti escludenti lo zero). **Da rieseguire con protocollo congelato** |
| M1-L (M1 + loss CC+KLD, pesi 0.5/0.5) | Pesi selezionati tramite tuning su subset di sviluppo (3 configurazioni testate). Risultato pre-freeze: CC 0.880, SIM 0.782, KLD 0.190. **Da rieseguire con protocollo congelato** |
| G (Adaptive Center Prior) | Implementato (`src/models/adaptive_center_prior.py`), base M1-L e prior B0 congelati durante il training del gate. Risultato pre-freeze: **negativo**, CC 0.863 (peggiore di M1-L su tutte e tre le metriche, bootstrap CI 95% conferma significativo). Diagnosticato con `scripts/diagnose_gate.py`: il gate converge verso un peso quasi-zero del center prior (α medio ≈0.0003, deviazione standard ≈0.0002 su 2500 immagini), stabile rispetto a due inizializzazioni diverse — non un bug, il prior adattivo non trova beneficio quando la base domina così uniformemente. **Da rieseguire con protocollo congelato** prima di considerarlo il risultato finale |
| M2 (encoder PVTv2-B1 pretrained + decoder multi-scala, loss CC+KLD 0.5/0.5) | Integrato in factory/trainer/evaluator. Risultato di **sviluppo** (subset 6k, 3 epoche): CC 0.892, SIM 0.789, KLD 0.177 — non ancora confrontabile con gli altri modelli (protocollo diverso). **Da esguire con protocollo congelato** |
| Confronto per immagine | `scripts/compare_evaluations.py` allinea i CSV tramite `image_id`; `scripts/bootstrap_ci.py` calcola intervalli di confidenza sui delta appaiati |
| Figure qualitative | `scripts/qualitative_figures.py` (script dedicato, non più solo celle notebook): confronto Immagine/GT/B0/B1/M1/M1-L/G/M2, selezione automatica di 5 esempi rappresentativi, α di G mostrato in etichetta |
| Report | Ancora da produrre; la catena B0→G è già una storia completa e argomentata (vedi sopra), utilizzabile per la sezione Results una volta rigenerata con protocollo congelato |

I risultati sopra etichettati "pre-freeze" **non sono i risultati ufficiali finali**: sono stati prodotti con `epochs: 3`, insufficiente perché l'early stopping (patience 5) possa mai attivarsi davvero — B1 e M1 in particolare erano ancora in netto miglioramento all'ultima epoca disponibile. Restano validi come **confronti relativi** (stesso protocollo per ogni coppia), non come numeri assoluti definitivi. Non riportare come risultato ufficiale un checkpoint ottenuto su un sottoinsieme, con configurazione non congelata, o prima del freeze.

## Protocollo di freeze (deciso il 24/09/2026)

Prima dei run finali, questi valori sono **congelati e uguali per tutti i modelli trainabili** (B1, M1, M1-L, G, M2) — nessuna eccezione, incluso per M2, il cui encoder Transformer pretrained potrebbe convergere più lentamente di un encoder interamente fine-tunato:

| Parametro | Valore |
| --- | --- |
| Max epochs | **30** (non 3: serve margine reale perché l'early stopping possa attivarsi) |
| Early stopping patience | **5** (invariata, già usata nel tuning di M1-L) |
| Selection metric | `cc`, `max` |
| Train split per i run finali | Completo (10.000), nessun `--dev_subset` |
| Checkpoint | Cartelle **pulite**, mai le stesse dei run di sviluppo — `train.py` riprende automaticamente da `<MODEL>_last.pt`, una cartella non vuota rischia un resume dal protocollo sbagliato |
| `internal_test` | Resta chiuso fino al freeze completo di codice, split, seed e iperparametri |

Ordine consigliato dei run finali: B1 → M1 → M1-L → G (richiede il checkpoint finale di M1-L come base) → M2 (indipendente, eseguibile in parallelo).

## Disegno sperimentale

```text
B0   center prior (senza training neurale)
B1   ResNet18 + decoder, MSE
M1   B1 + feature multi-scala (C3/C4/C5), stessa MSE
M1-L stessa architettura di M1, loss CC+KLD (pesi 0.5/0.5, scelti su tuning)
G    M1-L + prior centrale adattivo (base M1-L e B0 congelati, si allena solo il gate)
M2   encoder PVTv2-B1 pretrained (via timm) + decoder multi-scala comparabile a M1-L
```

I confronti principali sono `B1−B0`, `M1−B1`, `M1-L−M1`, `G−M1-L` e `M2−M1-L`. La configurazione corrente è in `configs/experiments.yaml`.

**Nota:** l'intestazione di `configs/experiments.yaml` contiene ancora un commento non aggiornato che dichiara `cc_weight`/`kld_weight` null e M1-L/G/M2 da non lanciare — i valori reali sono già 0.5/0.5 e tutti e tre i modelli sono stati lanciati con successo. Il commento va corretto insieme al freeze, senza cambiare il comportamento del codice.

## Protocollo dei dati

Il manifest condiviso definisce `train` (10.000), `tuning` (2.500) e `internal_test` (2.500). Non rigenerare lo split nei singoli esperimenti. Il dataset viene preparato in `/content/data_local` su Colab; checkpoint e risultati importanti vanno su Google Drive perché il disco della VM è temporaneo.

`SaliconDataset` restituisce immagine, `density_map_raw`, `density_map_prob`, `fixation_path` e `image_id`. B1/M1 usano la mappa raw come target MSE; M1-L/G/M2 usano la mappa probabilistica con loss CC+KLD; B0 usa la mappa probabilistica per il proprio prior. Durante sviluppo, selezione dei checkpoint e tuning si usa **solo `tuning`**. `internal_test` rimane chiuso fino al freeze del protocollo e al confronto finale.

Le metriche principali dell'evaluator sono **CC, SIM e KLD** (`KLD(target || prediction)`). NSS e sAUC sono implementate e testate, mantenute nel codice come possibile approfondimento, ma non fanno parte della tabella principale del report per decisione del 24/09.

## Notebook-regista Colab

Aprire `notebooks/colab_bootstrap.ipynb`. Il notebook monta Drive, aggiorna il repository, prepara SALICON e **chiama gli script del repository**: non contiene una seconda implementazione del training. La sezione 7 comprende:

1. preflight con la suite dei test;
2. training per modello (B0, B1, M1, M1-L, G, M2), disattivato di default finché non si decide il run — G richiede i checkpoint finali di M1-L e B0 come `--base_checkpoint`/`--center_prior_checkpoint`;
3. evaluation su `tuning` per tutti e sei i modelli, con CSV per immagine e JSON;
4. confronto dei CSV allineato per `image_id` sulle coppie congelate (`B1-B0`, `M1-B1`, `M1-L-M1`, `G-M1-L`, `M2-M1-L`) e bootstrap CI sui delta;
5. figura qualitativa tramite `scripts/qualitative_figures.py`: Immagine | Ground truth | B0 | B1 | M1 | M1-L | G | M2, selezione automatica di 5 esempi (α di G mostrato in etichetta);
6. sezione `internal_test` separata, bloccata di default: va aperta solo dopo il freeze definitivo.

Le celle operative sono disattivate di default (flag `RUN_* = False`). Prima di avviare run finali, verificare commit, YAML, split, seed, disponibilità dei checkpoint e spazio su Drive, e che le cartelle di destinazione siano pulite.

## Comandi essenziali

Da root del repository, dopo aver installato `requirements-colab.txt` e preparato il dataset:

```bash
python -m pytest -q
python scripts/smoke_test_m1.py
```

Training (protocollo da `configs/experiments.yaml`, congelato a epochs=30/patience=5 per i run finali):

```bash
python scripts/train.py --experiment B0 --data_dir /content/data_local --checkpoint_dir <cartella_pulita>
python scripts/train.py --experiment B1 --data_dir /content/data_local --checkpoint_dir <cartella_pulita>
python scripts/train.py --experiment M1 --data_dir /content/data_local --checkpoint_dir <cartella_pulita>
python scripts/train.py --experiment M1-L --data_dir /content/data_local --checkpoint_dir <cartella_pulita>
python scripts/train.py --experiment G --data_dir /content/data_local --checkpoint_dir <cartella_pulita> \
    --base_checkpoint <M1-L_best.pt finale> --center_prior_checkpoint <B0_center_map.pt>
python scripts/train.py --experiment M2 --data_dir /content/data_local --checkpoint_dir <cartella_pulita>
```

Diagnostica del gate di G (verifica se il peso α varia per immagine o è collassato a un valore costante):

```bash
python scripts/diagnose_gate.py --checkpoint <G_best.pt> --data_dir /content/data_local
```

Evaluation su `tuning`:

```bash
python scripts/evaluate.py --experiment M1 --checkpoint_path <M1_best.pt> --split tuning --data_dir /content/data_local --results_dir <cartella_evaluation>
```

L'evaluator salva `tuning_per_image.csv` e `tuning_summary.json` sotto la cartella del modello; il JSON include `n_samples`. Il checkpoint deve dichiarare l'`experiment` corretto (es. `M1-L`, `G`), altrimenti viene rifiutato. `internal_test` richiede il flag esplicito `--final_evaluation`, da usare **solo dopo il freeze**.

Confronto per immagine, con ID mancanti o duplicati trattati come errore:

```bash
python scripts/compare_evaluations.py \
  <cartella_evaluation>/B1/tuning_per_image.csv \
  <cartella_evaluation>/M1/tuning_per_image.csv \
  --left-name B1 --right-name M1 \
  --output <cartella_evaluation>/comparisons/M1_minus_B1.csv
```

Bootstrap CI (paired, 1000 resample) sui delta prodotti dal confronto:

```bash
python scripts/bootstrap_ci.py <cartella_evaluation>/comparisons/M1_minus_B1.csv \
    --n_resamples 1000 --seed 42 \
    --output <cartella_evaluation>/comparisons/bootstrap_M1_minus_B1.json
```

Se l'intervallo di confidenza al 95% non include lo zero, il delta è statisticamente robusto rispetto al rumore campionario tra le immagini del tuning set, non solo osservato.

Figura qualitativa (richiede i checkpoint di tutti e sei i modelli):

```bash
python scripts/qualitative_figures.py \
    --data_dir /content/data_local \
    --checkpoint_dir <cartella_pulita> \
    --results_dir <cartella_evaluation> \
    --output <percorso_figura.png>
```

## Prossimi passi

1. Eseguire i run finali con protocollo congelato (epochs=30, patience=5, cartelle pulite) per B1, M1, M1-L, G, M2, nell'ordine indicato sopra.
2. Valutare tutti e sei i modelli su `tuning`, produrre confronti per immagine e bootstrap CI sulle coppie congelate.
3. Generare la figura qualitativa finale con `scripts/qualitative_figures.py`.
4. Congelare definitivamente codice, configurazioni e criteri di selezione.
5. Valutare una sola volta su `internal_test`, produrre tabella, figure ufficiali e report.

Non caricare nel repository dataset SALICON, credenziali Kaggle, checkpoint grandi o output temporanei. Conservare i checkpoint e i risultati dei run su Drive.
