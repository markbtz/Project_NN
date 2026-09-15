# NNDL Saliency Project — Team [A/B/C]

Progetto di **visual saliency prediction (fixation prediction)** sul dataset **SALICON**.

Il progetto segue il piano sperimentale definito in `piano_sviluppo_2settimane_v2-1.pdf` (documento esterno al repository).

L'obiettivo è costruire e confrontare progressivamente diversi modelli di saliency prediction mantenendo invariato il protocollo sperimentale, in modo che ogni confronto modifichi una sola componente alla volta.

---

## Disegno sperimentale

La sequenza degli esperimenti, definita anche in `configs/experiments.yaml`, è:

```text
B0
Center Prior
    ↓
B1
ResNet18 + decoder
Loss: MSE
    ↓
M1
B1 + skip connections / feature multi-scala C3-C4-C5
Loss: stessa MSE di B1
    ↓
M1-L
Stessa architettura di M1
Loss: CC-loss + KLD
    ↓
G
M1-L + Adaptive Center Prior
    ↓
M2
Encoder Transformer gerarchico leggero
+ stesso decoder multi-scala
```

Il principio fondamentale è:

> **un solo cambiamento alla volta tra due esperimenti consecutivi**

In particolare:

- `B1 → M1` misura l'effetto delle skip connection / informazione multi-scala;
- `M1 → M1-L` misura l'effetto del cambio di loss;
- `M1-L → G` misura l'effetto dell'Adaptive Center Prior;
- `M1-L → M2` permette di studiare l'effetto del cambio di encoder.

Non bisogna cambiare contemporaneamente architettura e loss nello stesso confronto.

---

# Stato attuale del progetto

Le fasi di **setup**, **data pipeline**, **preprocessing**, **fixation pipeline**, **metriche** e **collegamento Dataset → training** sono state completate e verificate.

```text
SETUP
Repository GitHub                  ✅
Collaborazione GitHub              ✅
VS Code                            ✅
Google Colab                       ✅
GPU NVIDIA                         ✅
Google Drive                       ✅
Kaggle API                         ✅

DATA PIPELINE
Download SALICON                   ✅
Audit SALICON                      ✅
Fixation files disponibili         ✅
Split riproducibile                ✅
split_manifest.csv                 ✅
SaliconDataset PyTorch             ✅
Preprocessing RGB                  ✅
density_map_raw                    ✅
density_map_prob                   ✅
Horizontal flip sincronizzato      ✅
DataLoader                         ✅
Smoke test                         ✅

FIXATION PIPELINE
Parsing file .mat SALICON          ✅
Aggregazione fixation osservatori  ✅
Coordinate 1-based → 0-based       ✅
Resize fixation a 256×192          ✅

METRICHE
CC                                 ✅
SIM                                ✅
KLD                                ✅
NSS                                ✅
sAUC                               ✅
Unit test metriche                 ✅ 12 passed
Sanity check NSS su SALICON        ✅
Sanity check sAUC su SALICON       ✅

TRAINING INTEGRATION
train.py → SaliconDataset           ✅
DummyDensityDataset rimosso         ✅
B1 target = density_map_raw         ✅
B0 target = density_map_prob        ✅
Batch SALICON reale verificato      ✅
B1 forward/backward end-to-end      ⏳
```

Il prossimo blocco di lavoro riguarda le **loss**, seguito da B0 e B1.

---

# Struttura del repository

```text
Project_NN/
│
├── configs/
│   ├── data.yaml
│   └── experiments.yaml
│
├── notebooks/
│   └── colab_bootstrap.ipynb
│
├── results/
│   └── split_manifest.csv
│
├── scripts/
│   ├── audit_dataset.py
│   ├── download_salicon.py
│   ├── smoke_test.py
│   └── train.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── fixations.py
│   │   └── splits.py
│   │
│   ├── losses/
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline.py
│   │
│   └── metrics.py
│
├── tests/
│   └── test_metrics.py
│
├── .gitignore
├── README.md
├── environment-mac.yml
└── requirements-colab.txt
```

Le cartelle `src/losses/`, `src/models/` e `tests/` verranno completate progressivamente durante le fasi successive.

---

# Workflow di sviluppo

Il workflow utilizzato dal team è:

```text
VS Code
   ↓
sviluppo / modifica codice
   ↓
GitHub Desktop
   ↓
Commit + Push
   ↓
GitHub
   ↓
git pull
   ↓
Google Colab
   ↓
test / training / evaluation
```

## VS Code

Utilizzato principalmente per:

- sviluppo del codice;
- modifica delle configurazioni;
- modifica del README;
- implementazione di Dataset, metriche, loss, modelli e script.

## GitHub Desktop

Utilizzato per:

- controllare i file modificati;
- creare i commit;
- eseguire push verso GitHub;
- ricevere gli aggiornamenti del repository.

## Google Colab

Utilizzato principalmente per:

- test della pipeline;
- training;
- evaluation;
- utilizzo GPU quando necessario;
- gestione dei checkpoint;
- sanity check su dati reali SALICON.

Le celle temporanee di Colab vengono usate solo per eseguire o verificare il codice presente nel repository. La logica permanente deve restare nei file Python del progetto.

---

# Setup ambiente

## Mac / sviluppo locale

È disponibile:

```text
environment-mac.yml
```

Creazione ambiente:

```bash
conda env create -f environment-mac.yml
conda activate nndl-saliency
```

Il Mac viene utilizzato principalmente per sviluppo, debug leggero e controllo del codice.

## Google Colab

Per training e test pesanti utilizzare:

```text
notebooks/colab_bootstrap.ipynb
```

Le dipendenze Colab sono definite in:

```text
requirements-colab.txt
```

Installazione manuale:

```bash
pip install -r requirements-colab.txt
```

Per test di Dataset, DataLoader, metriche e smoke test non è necessaria la GPU. La GPU verrà utilizzata soprattutto durante il training dei modelli neurali.

---

# Dataset SALICON

Il progetto utilizza **SALICON**.

Il dataset viene scaricato tramite Kaggle utilizzando:

```text
scripts/download_salicon.py
```

La struttura logica attesa è:

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── maps/
│   ├── train/
│   └── val/
│
└── fixations/
    ├── train/
    ├── val/
    └── test/
```

Su Colab il dataset viene utilizzato in:

```text
/content/data_local
```

---

# Audit del dataset

Lo script:

```text
scripts/audit_dataset.py
```

è stato utilizzato per controllare struttura e consistenza del dataset.

Risultato dell'audit:

```text
images                   : 20000
density_maps             : 15000
fixation_files           : 20000
```

Sono inoltre state trovate:

```text
Coppie immagine ↔ density map : 15000
Immagini senza density map    : 5000
Mappe senza immagine          : 0
```

Le 5000 immagini prive di density map appartengono al **test ufficiale SALICON**, per il quale non viene utilizzata una ground truth pubblica nel protocollo interno del progetto.

L'audit ha inoltre confermato la presenza dei file di fixation `.mat`.

Esecuzione:

```bash
python scripts/audit_dataset.py --data_dir /content/data_local
```

---

# Setup Kaggle API

Per il download iniziale di SALICON è necessario un account Kaggle.

Nel progetto viene utilizzato:

```text
kaggle.json
```

Procedura:

1. accedere a Kaggle;
2. aprire `Settings`;
3. creare una **Legacy API Key**;
4. scaricare `kaggle.json`.

Il file contiene credenziali private.

**`kaggle.json` non deve mai essere caricato su GitHub.**

---

# Gestione del dataset su Google Colab

Leggere decine di migliaia di piccoli file direttamente da Google Drive è lento.

Per questo motivo il training non deve leggere direttamente il dataset dal Drive montato.

La strategia adottata è:

```text
Google Drive
     │
     │ salicon.tar
     ↓
Colab VM
     │
     ↓
/content/data_local
```

Su Google Drive viene conservato:

```text
/content/drive/MyDrive/nndl-saliency/salicon.tar
```

A ogni nuova sessione Colab il dataset viene copiato/estratto sul disco locale della VM.

## Technical debt noto sul layout del dataset

In alcune estrazioni dell'archivio è stata osservata una struttura legacy:

```text
images/images/train
images/images/val
images/images/test
```

mentre il layout atteso dal Dataset è:

```text
images/train
images/val
images/test
```

La correzione definitiva di `salicon.tar` / della cella `6bis` del notebook è ancora un piccolo **technical debt non bloccante**.

Nelle sessioni correnti la struttura viene corretta prima dell'esecuzione degli script.

---

# Configurazione dataset

La configurazione principale si trova in:

```text
configs/data.yaml
```

Configurazione di riferimento:

```yaml
dataset_root: "data/"

images_dir: "images"
maps_dir: "maps"
fixations_dir: "fixations"

has_fixation_coordinates: true

colab_local_cache_dir: "/content/data_local"

seed: 42

n_train: 10000
n_tuning: 2500
n_internal_test: 2500

dev_subset_n_train: 6000

input_size: [256, 192]

density_map_epsilon: 1.0e-6

augmentation: "hflip_sync"
```

Non devono essere inseriti nel codice path personali del tipo:

```text
C:\Users\NomeUtente\...
/Users/nomeutente/...
```

I path condivisi devono essere configurabili.

---

# Split SALICON

Lo split utilizzato dal progetto è fisso e riproducibile:

```text
Training       : 10000
Tuning         : 2500
Internal test  : 2500
```

Le 10000 immagini `train` ufficiali SALICON vengono usate come training set.

Le 5000 immagini della validation ufficiale SALICON vengono divise in:

```text
2500 tuning
2500 internal test
```

con:

```text
seed = 42
```

Lo split definitivo è salvato in:

```text
results/split_manifest.csv
```

Esempio:

```text
image_id,official_split,split
COCO_train2014_000000000009,train,train
COCO_train2014_000000000025,train,train
...
COCO_val2014_XXXXXXXXXXXX,val,tuning
...
COCO_val2014_XXXXXXXXXXXX,val,internal_test
```

**Il manifest non deve essere rigenerato indipendentemente dai singoli collaboratori.**

Tutti gli esperimenti devono usare esattamente lo stesso split.

La logica di generazione è contenuta in:

```text
src/data/splits.py
```

---

# Dataset PyTorch

La pipeline PyTorch è implementata in:

```text
src/data/dataset.py
```

La classe principale è:

```python
SaliconDataset
```

Il Dataset legge il manifest e carica:

```text
immagine
+
density map
+
fixation file
```

Ogni campione restituisce:

```python
{
    "image": image_tensor,
    "density_map_raw": density_map_raw,
    "density_map_prob": density_map_prob,
    "fixation_path": "...",
    "image_id": "...",
    "split": "..."
}
```

Sono mantenute due rappresentazioni della stessa density map per separare chiaramente gli usi basati su intensità da quelli probabilistici.

---

# Preprocessing immagini

Ogni immagine viene:

1. caricata in RGB;
2. ridimensionata a `256 × 192`;
3. convertita in tensore PyTorch;
4. normalizzata utilizzando le statistiche ImageNet.

Valori:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Shape finale:

```text
[3, 192, 256]
```

---

# Preprocessing density map

Ogni density map viene:

1. caricata in grayscale;
2. ridimensionata a `256 × 192` tramite interpolazione bilineare;
3. convertita in `float32`;
4. trasformata in due rappresentazioni differenti.

## `density_map_raw`

I valori originali grayscale `0...255` vengono convertiti nell'intervallo:

```text
[0, 1]
```

tramite:

```text
density_map_raw = density_map / 255
```

La shape finale è:

```text
[1, 192, 256]
```

Questa rappresentazione:

- non è normalizzata a somma 1;
- viene usata principalmente come target della MSE;
- verrà quindi usata da B1 e M1.

## `density_map_prob`

A partire dalla raw map viene costruita una distribuzione spaziale:

```text
density_map_prob =
(density_map_raw + ε)
/
Σ(density_map_raw + ε)
```

con:

```text
ε = 1e-6
```

La shape finale è:

```text
[1, 192, 256]
```

e vale:

```text
Σ density_map_prob ≈ 1
```

Questa rappresentazione viene usata per:

```text
B0
SIM
KLD
operazioni probabilistiche
```

La separazione tra `raw` e `prob` evita di usare una target map normalizzata a somma 1 direttamente con MSE.

---

# Data augmentation

Durante il training viene utilizzato:

```text
horizontal flip
```

Il flip viene applicato in modo sincronizzato a:

```text
image
density_map_raw
density_map_prob
```

In questo modo tutte le rappresentazioni rimangono spazialmente allineate.

Non vengono utilizzati random crop o trasformazioni geometriche aggressive perché potrebbero alterare artificialmente la distribuzione spaziale della salienza.

---

# Fixation data

I fixation file SALICON sono disponibili in formato:

```text
.mat
```

L'ispezione dei file ha mostrato la struttura principale:

```text
image
resolution
gaze
```

Il campo `gaze` contiene:

```text
location
timestamp
fixations
```

Le fixation sono memorizzate per osservatore come array:

```text
[N, 2]
```

con coordinate:

```text
[x, y]
```

È stato verificato su 500 file che le coordinate SALICON sono **1-based**:

```text
x ∈ [1, 640]
y ∈ [1, 480]
```

La pipeline converte quindi:

```text
SALICON 1-based
      ↓
Python/PyTorch 0-based
      ↓
resize 640×480 → 256×192
```

La logica permanente è implementata in:

```text
src/data/fixations.py
```

Il parser:

- legge il file `.mat`;
- estrae `gaze["fixations"]`;
- aggrega le fixation dei diversi osservatori;
- legge la risoluzione originale;
- converte le coordinate da 1-based a 0-based;
- ridimensiona le coordinate alla risoluzione usata dal modello.

Il parser è SALICON-specifico; le metriche rimangono invece il più possibile indipendenti dal dataset.

---

# DataLoader

La pipeline utilizza:

```python
torch.utils.data.DataLoader
```

È stato verificato con successo un batch reale SALICON.

Shape osservate:

```text
Images:
torch.Size([4, 3, 192, 256])

Raw targets:
torch.Size([4, 1, 192, 256])

Probability targets:
torch.Size([4, 1, 192, 256])
```

Sono state inoltre verificate le proprietà:

```text
density_map_raw:
min >= 0
max <= 1

density_map_prob:
somma per ogni campione ≈ 1
```

Output osservato:

```text
Raw range: 0.0 1.0

Probability sums:
tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

Questo conferma che il contratto:

```text
SaliconDataset
      ↓
DataLoader
      ↓
training
```

è compatibile con dati SALICON reali.

---

# Smoke test

È disponibile:

```text
scripts/smoke_test.py
```

Esecuzione:

```bash
python scripts/smoke_test.py
```

Lo smoke test controlla:

- `train = 10000`;
- `tuning = 2500`;
- `internal_test = 2500`;
- caricamento corretto dei tre split;
- shape delle immagini;
- shape di `density_map_raw`;
- shape di `density_map_prob`;
- assenza di NaN e Inf;
- valori di `density_map_raw` nell'intervallo `[0,1]`;
- somma di `density_map_prob ≈ 1`;
- presenza dei fixation file;
- corretto funzionamento del DataLoader;
- corretto batching delle due rappresentazioni.

Risultato verificato:

```text
============================================================
SMOKE TEST PASSED
============================================================
```

---

# Integrazione con il training

Lo script:

```text
scripts/train.py
```

era stato inizialmente sviluppato utilizzando un `DummyDensityDataset` con tensori casuali per verificare forward, backward e checkpoint senza dipendere dalla pipeline dati reale.

Ora è collegato direttamente a:

```python
SaliconDataset
```

Il dataset fittizio non viene più usato nella pipeline reale.

## B1

Per B1 il training loop usa:

```python
images = batch["image"]
targets = batch["density_map_raw"]
```

perché B1 utilizza:

```text
MSE
```

La parte già presente relativa a:

```text
B1Baseline
AdamW
checkpoint
best checkpoint
resume
```

è stata mantenuta.

## B0

B0 non usa backpropagation.

Per costruire il center prior vengono utilizzate:

```python
batch["density_map_prob"]
```

Il center prior viene calcolato come media delle density map probabilistiche del training set.

## Stato dell'integrazione

È stato verificato un batch SALICON reale end-to-end fino all'ingresso del training:

```text
SALICON
   ↓
SaliconDataset
   ↓
DataLoader
   ↓
train.py
```

Il training completo di B1 non è ancora stato eseguito.

Prima del training vero verranno completate e testate le loss del progetto e verrà eseguito un mini test forward/backward end-to-end.

---

# Metriche di valutazione

Le metriche sono implementate in:

```text
src/metrics.py
```

Sono disponibili:

```text
CC
SIM
KLD
NSS
sAUC
```

I test automatici sono contenuti in:

```text
tests/test_metrics.py
```

Risultato:

```text
12 passed
```

## CC — Correlation Coefficient

Misura la correlazione lineare tra prediction e target.

```text
+1  correlazione perfetta
 0  nessuna correlazione lineare
-1  correlazione inversa
```

**Più alto è meglio.**

## SIM — Similarity

Le mappe vengono normalizzate come distribuzioni e viene calcolata la loro sovrapposizione:

```text
SIM = Σ min(P, Q)
```

Range:

```text
0 → nessuna sovrapposizione
1 → distribuzioni identiche
```

**Più alto è meglio.**

## KLD — Kullback-Leibler Divergence

Nel progetto viene utilizzata esplicitamente la convenzione:

```text
KLD(target || prediction)
```

Il valore ideale è:

```text
0
```

**Più basso è meglio.**

La stessa convenzione deve essere mantenuta in tutti gli esperimenti.

## NSS — Normalized Scanpath Saliency

NSS usa le fixation reali degli osservatori.

La saliency map viene standardizzata:

```text
S_norm = (S - mean(S)) / std(S)
```

e viene poi calcolata la media della saliency standardizzata nei punti fissati.

Input:

```text
saliency prediction
+
fixation della stessa immagine
```

**Più alto è meglio.**

È stato eseguito con successo un sanity check end-to-end su una vera density map SALICON e sulle relative fixation.

Il valore del sanity check non rappresenta la performance di un modello, perché la ground-truth density map è stata usata temporaneamente come prediction.

## sAUC — Shuffled AUC

sAUC utilizza:

```text
prediction
+
fixation positive della stessa immagine
+
fixation negative provenienti da altre immagini
```

Range:

```text
0.0 → separazione pessima
0.5 → comportamento casuale
1.0 → separazione perfetta
```

**Più alto è meglio.**

È stato eseguito con successo un sanity check end-to-end su SALICON.

Per il sanity check sono state utilizzate fixation negative provenienti da una seconda immagine.

Il protocollo definitivo di evaluation dovrà fissare in modo riproducibile il campionamento delle fixation negative da più immagini.

---

# Test delle metriche

Esecuzione:

```bash
pytest -q tests/test_metrics.py
```

Output verificato:

```text
12 passed
```

I test controllano, tra le altre cose:

```text
Mappe identiche:
CC  ≈ 1
SIM ≈ 1
KLD ≈ 0

NSS:
fixation su regione saliente      → valore positivo
fixation su regione non saliente  → valore negativo
mappa costante                    → NSS = 0

sAUC:
separazione perfetta              → 1
separazione invertita             → 0
parità completa                   → 0.5
```

---

# Generalità del codice rispetto al dataset

Non tutto il progetto è dataset-agnostic.

## Componenti generici

Sono progettati per essere riutilizzabili:

```text
src/metrics.py
CC
SIM
KLD
NSS
sAUC
```

NSS e sAUC ricevono coordinate già convertite nel formato `[x, y]` e non dipendono direttamente dal formato `.mat` di SALICON.

Anche la logica generale di training ed evaluation sarà mantenuta il più possibile indipendente dal dataset.

## Componenti SALICON-specifici

Sono invece specifici del dataset:

```text
SaliconDataset
src/data/fixations.py
src/data/splits.py
scripts/audit_dataset.py
```

Un altro dataset richiederebbe un proprio loader/parser mantenendo invariata, per quanto possibile, la parte generica del progetto.

---

# Loss — prossimo blocco di sviluppo

Il prossimo blocco riguarda le loss utilizzate durante il training.

Verranno implementate in:

```text
src/losses/
```

e testate in:

```text
tests/test_losses.py
```

Le loss previste sono:

```text
MSE
CC-loss
KLD loss
CC-loss + KLD
```

## Convenzione dei target

```text
MSE
→ density_map_raw

KLD
→ density_map_prob
```

Per `CC-loss` verrà mantenuto un protocollo coerente con l'implementazione della metrica CC.

La loss combinata prevista per M1-L sarà basata su:

```text
CC-loss + KLD
```

---

# B0 — Center Prior

Dopo la chiusura del blocco loss verrà completato B0.

B0 non richiede una rete neurale né backpropagation.

La predizione verrà ottenuta dalla media delle `density_map_prob` del training set:

```text
density_map_prob train 1
density_map_prob train 2
...
density_map_prob train N
        ↓
       media
        ↓
   CENTER PRIOR
```

B0 serve a quantificare quanto del problema possa essere spiegato esclusivamente dal **center bias**.

Il prior ottenuto verrà riutilizzato successivamente nel modello:

```text
G — Adaptive Center Prior
```

---

# B1 — Baseline neurale

Dopo B0 verrà addestrato B1:

```text
ResNet18 pretrained
+
decoder semplice
+
MSE
+
density_map_raw
```

B1 rappresenterà la baseline neurale principale.

---

# M1 — Multi-scale

M1 aggiungerà feature multi-scala provenienti da:

```text
C3
C4
C5
```

tramite skip connections.

La loss resterà la stessa di B1:

```text
MSE
```

Il confronto:

```text
B1 → M1
```

isolerà l'effetto della modifica architetturale.

---

# M1-L — Cambio della loss

M1-L manterrà esattamente la stessa architettura di M1.

Cambierà soltanto la funzione obiettivo:

```text
CC-loss + KLD
```

Il confronto:

```text
M1 → M1-L
```

misurerà l'effetto del cambio di loss.

---

# G — Adaptive Center Prior

G rappresenta la componente originale principale del progetto.

Concettualmente:

```text
S(x) =
(1 - α(x)) * S_M1-L(x)
+
α(x) * P_center
```

dove:

```text
P_center
```

è il prior ottenuto da B0.

---

# M2 — Advanced model

M2 utilizzerà un encoder Transformer gerarchico leggero.

Le opzioni considerate includono backbone gerarchici disponibili tramite `timm`, ad esempio:

```text
Swin
PVT
```

con interfaccia:

```python
features_only=True
```

L'obiettivo è mantenere il decoder comparabile con M1/M1-L e modificare principalmente l'encoder.

---

# Protocollo di evaluation

Tutti i modelli devono utilizzare:

```text
stesso preprocessing
stesso split
stesso internal test
stesse metriche
stessa convenzione KLD
stesso protocollo per NSS/sAUC
```

Le metriche implementate sono:

```text
CC
SIM
KLD
NSS
sAUC
```

Il protocollo definitivo di sAUC dovrà specificare in modo riproducibile la selezione delle fixation negative.

---

# Bootstrap Confidence Intervals

Nella fase finale verranno calcolati bootstrap confidence intervals sui confronti principali.

Ad esempio:

```text
B1 vs M1
M1 vs M1-L
M1-L vs G
M1-L vs M2
```

Il bootstrap verrà effettuato sulle predizioni già ottenute e non richiederà nuovi training.

---

# Collaborazione

Prima di iniziare una sessione di sviluppo:

```text
Pull / git pull
```

Workflow:

```text
aggiornamento repository
        ↓
sviluppo
        ↓
test
        ↓
commit
        ↓
push
```

È importante evitare di modificare contemporaneamente gli stessi file quando possibile.

---

# Convenzioni

## Split

Lo split ufficiale è:

```text
results/split_manifest.csv
```

Non deve essere rigenerato individualmente.

## Dataset

Il dataset SALICON non deve essere caricato su GitHub.

## Configurazioni

Le configurazioni condivise devono stare in:

```text
configs/
```

Gli iperparametri importanti dovrebbero essere definiti in configurazione invece di essere hardcoded negli script quando possibile.

## Path

Non utilizzare path personali nel codice condiviso.

Il path:

```text
/content/data_local
```

è specifico della VM Colab.

---

# File che NON devono essere caricati su GitHub

Non committare:

```text
dataset SALICON
*.jpg del dataset
*.png del dataset
*.mat del dataset
salicon.tar
checkpoint pesanti
predizioni pesanti
kaggle.json
.env
API key
password
credenziali
```

In particolare:

```text
kaggle.json
```

contiene credenziali Kaggle private.

---

# Checkpoint

I checkpoint dei modelli vengono salvati su:

```text
Google Drive
```

e non nel repository.

Questo permette di mantenere i training persistenti anche quando una sessione Colab termina.

`train.py` mantiene la logica di:

```text
last checkpoint
best checkpoint
resume
```

già presente nella baseline iniziale.

---

# Riproducibilità

Il progetto cerca di garantire la riproducibilità tramite:

```text
seed fisso
split fisso
manifest condiviso
configurazioni versionate
preprocessing condiviso
metriche comuni
protocollo di evaluation condiviso
checkpoint
```

Seed corrente:

```text
42
```

---

# Comandi utili

## Aggiornare repository in Colab

```bash
git pull
```

## Installare dipendenze

```bash
pip install -r requirements-colab.txt
```

## Audit dataset

```bash
python scripts/audit_dataset.py --data_dir /content/data_local
```

## Smoke test

```bash
python scripts/smoke_test.py
```

Output corretto:

```text
SMOKE TEST PASSED
```

## Test metriche

```bash
pytest -q tests/test_metrics.py
```

Output attuale:

```text
12 passed
```

## Training B1

Quando verrà avviato il training reale:

```bash
python scripts/train.py     --experiment B1     --epochs 3     --batch_size 8     --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
```

## Calcolo B0

```bash
python scripts/train.py     --experiment B0     --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
```

Questi comandi sono già collegati a `SaliconDataset`; B0/B1 non sono ancora considerati esperimenti completati.

---

# Entry point

Attualmente sono disponibili:

```text
scripts/audit_dataset.py
scripts/smoke_test.py
scripts/train.py
```

`train.py` è collegato al dataset reale SALICON e contiene il supporto iniziale per B0/B1.

Nelle fasi successive verrà aggiunto/completato:

```text
scripts/evaluate.py
```

Non è necessario concentrare tutta la logica in un singolo `main.py`.

L'obiettivo finale è avere entry point distinti e chiari per:

```text
smoke test
training
evaluation
```

---

# Roadmap sintetica

```text
FASE 1 — SETUP
✅ Repository
✅ Ambiente
✅ Colab
✅ GPU
✅ Google Drive
✅ Kaggle

FASE 2 — DATA PIPELINE
✅ SALICON
✅ Audit
✅ Split
✅ Manifest
✅ SaliconDataset
✅ density_map_raw
✅ density_map_prob
✅ Preprocessing
✅ Data augmentation
✅ DataLoader
✅ Smoke test

FASE 3 — FIXATION PIPELINE
✅ Analisi struttura .mat
✅ Estrazione gaze["fixations"]
✅ Aggregazione osservatori
✅ Verifica coordinate 1-based
✅ Conversione 1-based → 0-based
✅ Resize fixation a 256×192

FASE 4 — METRICHE
✅ CC
✅ SIM
✅ KLD
✅ NSS
✅ sAUC
✅ 12 unit test
✅ NSS sanity check
✅ sAUC sanity check

FASE 5 — TRAINING INTEGRATION
✅ DummyDensityDataset sostituito
✅ train.py collegato a SaliconDataset
✅ B1 → density_map_raw
✅ B0 → density_map_prob
✅ batch SALICON reale verificato
⏳ forward/backward B1 end-to-end

FASE 6 — LOSS
⏳ MSE
⏳ CC-loss
⏳ KLD loss
⏳ CC + KLD
⏳ test loss

FASE 7 — BASELINE
⏳ B0 Center Prior
⏳ B1 ResNet18 + decoder

FASE 8 — MODELLI
⏳ M1
⏳ M1-L
⏳ G
⏳ M2

FASE 9 — TRAINING / EVALUATION
⏳ Training completo
⏳ Checkpoint/resume verificato
⏳ Evaluation completa
⏳ Protocollo sAUC definitivo
⏳ Bootstrap CI
⏳ Figure qualitative

FASE 10 — CONSEGNA
⏳ Riproducibilità da clone pulito
⏳ Report finale
```

---

# Prossimo obiettivo operativo

Il prossimo blocco da implementare è:

```text
LOSS
├── MSE
├── CC-loss
├── KLD loss
└── CC-loss + KLD
```

con relativi unit test.

Subito dopo:

```text
B0 — Center Prior
```

e successivamente:

```text
B1 — ResNet18 + decoder + MSE
```

Da quel momento inizierà il confronto sperimentale tra baseline e modelli neurali.
