# NNDL Saliency Project

Predizione di mappe di salienza su SALICON. Il progetto confronta i modelli cambiando una componente alla volta, con split e protocollo di valutazione condivisi.

## Stato del progetto

| Blocco | Stato al 22/09/2026 |
| --- | --- |
| Dataset, split e preprocessing | Implementati; manifest fisso in `results/split_manifest.csv` |
| Metriche e loss | CC/SIM/KLD e loss implementate e testate; NSS/sAUC implementate ma non integrate nell'evaluation principale |
| B0 (center prior) | Codice e caricamento nell'evaluator pronti; risultato finale da produrre |
| B1 (ResNet18 + decoder, MSE) | Codice, training, validation e checkpoint pronti; run finale da produrre |
| M1 (multi-scala, MSE) | Modello, factory, trainer ed evaluator integrati; run finale da produrre |
| M1-L, G, M2 | Ancora da completare e integrare nel trainer/evaluator |
| Confronto per immagine | `scripts/compare_evaluations.py` allinea i CSV tramite `image_id` |
| Figure e report | Figure semplici predisposte nel notebook; risultati ufficiali e report ancora da produrre |

I test e gli eventuali run brevi di sviluppo **non** sono risultati sperimentali finali. Non riportare come risultato ufficiale un checkpoint ottenuto su un sottoinsieme o con configurazione non congelata.

## Disegno sperimentale

```text
B0   center prior (senza training neurale)
B1   ResNet18 + decoder, MSE
M1   B1 + feature multi-scala, stessa MSE
M1-L stessa architettura di M1, loss CC+KLD
G    prior centrale adattivo; mantenere comparabile la base scelta
M2   encoder gerarchico leggero, decoder comparabile
```

I confronti principali sono `B1−B0`, `M1−B1`, `M1-L−M1` e l'effetto di G rispetto alla sua base. La configurazione corrente è in `configs/experiments.yaml`; M1-L/G/M2 non vanno lanciati finché implementazione e parametri non sono chiusi.

## Protocollo dei dati

Il manifest condiviso definisce `train` (10.000), `tuning` (2.500) e `internal_test` (2.500). Non rigenerare lo split nei singoli esperimenti. Il dataset viene preparato in `/content/data_local` su Colab; checkpoint e risultati importanti vanno su Google Drive perché il disco della VM è temporaneo.

`SaliconDataset` restituisce immagine, `density_map_raw`, `density_map_prob`, `fixation_path` e `image_id`. B1/M1 usano la mappa raw come target MSE; B0 usa la mappa probabilistica. Durante sviluppo, selezione dei checkpoint e tuning si usa **solo `tuning`**. `internal_test` rimane chiuso fino al freeze del protocollo e al confronto finale.

Le metriche principali dell'evaluator sono **CC, SIM e KLD** (`KLD(target || prediction)`). NSS e sAUC restano nel codice, ma sono opzionali per il report finale; il protocollo corrente è descritto in `scripts/docs/evaluation_protocol.md`.

## Notebook-regista Colab

Aprire `notebooks/colab_bootstrap.ipynb`. Il notebook monta Drive, aggiorna il repository, prepara SALICON e **chiama gli script del repository**: non contiene una seconda implementazione del training. La sezione 7 comprende:

1. preflight con la suite dei test;
2. training, disattivato di default finché non si decide il run;
3. evaluation su `tuning`, con CSV per immagine e JSON;
4. confronto dei CSV allineato per `image_id`;
5. figure `immagine | ground truth | B0 | B1 | M1` per il miglior/peggior CC di M1 e la variazione CC più favorevole rispetto a B1, selezionate dai CSV del `tuning` quando esistono checkpoint coerenti.

Le celle operative sono disattivate di default. Prima di avviare run finali, verificare commit, YAML, split, seed, disponibilità dei checkpoint e spazio su Drive. Quando il codice dei modelli successivi sarà integrato, aggiungerli alle liste del notebook solo dopo aver verificato che trainer ed evaluator li supportino.

## Comandi essenziali

Da root del repository, dopo aver installato `requirements-colab.txt` e preparato il dataset:

```bash
python -m pytest -q
python scripts/smoke_test_m1.py
```

Per i modelli attualmente supportati dal trainer:

```bash
python scripts/train.py --experiment B0 --data_dir /content/data_local --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
python scripts/train.py --experiment B1 --data_dir /content/data_local --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
python scripts/train.py --experiment M1 --data_dir /content/data_local --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
```

I valori di default (incluse le epoche) vengono da `configs/experiments.yaml`: **non sono automaticamente il protocollo finale**. Deciderli e congelarli prima dei run ufficiali.

Esempio di evaluation su `tuning`:

```bash
python scripts/evaluate.py --experiment M1 --checkpoint_path /content/drive/MyDrive/nndl-saliency/checkpoints/M1_best.pt --split tuning --data_dir /content/data_local --results_dir /content/drive/MyDrive/nndl-saliency/evaluation
```

L'evaluator salva `tuning_per_image.csv` e `tuning_summary.json` sotto la cartella del modello; il JSON include `n_samples`. Il checkpoint M1 deve dichiarare `experiment: M1`, altrimenti viene rifiutato. `internal_test` richiede il flag esplicito `--final_evaluation`, da usare **solo dopo il freeze**.

Confronto per immagine, con ID mancanti o duplicati trattati come errore:

```bash
python scripts/compare_evaluations.py \
  /content/drive/MyDrive/nndl-saliency/evaluation/B1/tuning_per_image.csv \
  /content/drive/MyDrive/nndl-saliency/evaluation/M1/tuning_per_image.csv \
  --left-name B1 --right-name M1 \
  --output /content/drive/MyDrive/nndl-saliency/evaluation/comparisons/M1_minus_B1.csv
```

Il CSV contiene differenze `right−left` per CC/SIM/KLD; il notebook stampa anche le differenze medie. Le figure del notebook sono qualitative: ciascuna mappa viene riscalata solo per la visualizzazione, quindi **non** sostituiscono le metriche. Un bootstrap appaiato con intervalli di confidenza è un possibile approfondimento per il report, non un prerequisito del codice né uno script obbligatorio adesso.

## Prossimi passi

1. Chiudere M1-L (loss e configurazione), G con ablation, poi valutare M2 in base al tempo disponibile.
2. Estendere trainer ed evaluator soltanto quando i nuovi modelli sono testati, senza cambiare retroattivamente il protocollo degli esperimenti già definiti.
3. Eseguire test, smoke test e mini-run di sviluppo sul `tuning`; verificare anche salvataggio e resume su Drive prima di affidarsi a run lunghi.
4. Congelare codice, configurazioni e criteri di selezione; eseguire i run completi dei modelli supportati.
5. Valutare una sola volta su `internal_test`, produrre tabella, figure ufficiali e report.

Non caricare nel repository dataset SALICON, credenziali Kaggle, checkpoint grandi o output temporanei. Conservare i checkpoint e i risultati dei run su Drive.
