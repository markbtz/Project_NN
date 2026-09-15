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

La parte di setup, download e preparazione della pipeline dati è stata completata.

Attualmente risultano funzionanti:

```text
Repository GitHub              ✅
Collaborazione GitHub          ✅
VS Code                        ✅
Google Colab                   ✅
GPU NVIDIA                     ✅
Google Drive                   ✅
Kaggle API                     ✅

Download SALICON               ✅
Audit SALICON                  ✅
Fixation files disponibili     ✅

Archivio salicon.tar           ✅
Cache locale Colab             ✅

Split riproducibile            ✅
split_manifest.csv             ✅

SaliconDataset PyTorch         ✅
Preprocessing                  ✅
Data augmentation              ✅
DataLoader                     ✅
Smoke test                     ✅
```

Il prossimo blocco di lavoro riguarda:

```text
Metriche CC / SIM / KLD
↓
B0
↓
B1
↓
M1
↓
M1-L
↓
G
↓
M2
```

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
│   └── smoke_test.py
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── splits.py
│   │
│   ├── losses/
│   └── models/
│
├── tests/
│
├── .gitignore
├── README.md
├── environment-mac.yml
└── requirements-colab.txt
```

Le cartelle `src/losses/`, `src/models/` e `tests/` verranno riempite progressivamente durante le prossime fasi.

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
test / training / evaluation con GPU
```

### VS Code

Utilizzato principalmente per:

- sviluppo del codice;
- modifica delle configurazioni;
- modifica del README;
- creazione di Dataset, modelli, loss e script.

### GitHub Desktop

Utilizzato per:

- controllare i file modificati;
- creare i commit;
- eseguire push verso GitHub;
- ricevere eventuali aggiornamenti del repository.

### Google Colab

Utilizzato principalmente per:

- GPU;
- test della pipeline;
- training;
- evaluation;
- gestione dei checkpoint.

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

---

## Google Colab

Per training e test pesanti utilizzare:

```text
notebooks/colab_bootstrap.ipynb
```

Il notebook si occupa della configurazione della sessione Colab.

Le dipendenze Colab sono definite in:

```text
requirements-colab.txt
```

Installazione manuale:

```bash
pip install -r requirements-colab.txt
```

---

# Dataset SALICON

Il progetto utilizza **SALICON**.

Il dataset viene scaricato tramite Kaggle utilizzando:

```text
scripts/download_salicon.py
```

La struttura attesa è:

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

---

# Audit del dataset

Lo script:

```text
scripts/audit_dataset.py
```

è stato utilizzato per controllare il dataset scaricato.

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

Le 5000 immagini prive di density map appartengono al **test ufficiale SALICON**, per il quale non viene utilizzata una ground truth pubblica nel nostro protocollo interno.

L'audit ha inoltre confermato la presenza dei file di fixation `.mat`.

Questo permette di considerare, oltre a:

```text
CC
SIM
KLD
```

anche metriche basate su fixation coordinates come:

```text
NSS
sAUC
```

nelle fasi successive.

---

# Eseguire nuovamente l'audit

Su Colab:

```bash
python scripts/audit_dataset.py --data_dir /content/data_local
```

---

# Setup Kaggle API

Per il download iniziale di SALICON è necessario un account Kaggle.

Nel progetto viene utilizzato il file legacy:

```text
kaggle.json
```

Procedura:

1. accedere a Kaggle;
2. aprire `Settings`;
3. nella sezione API creare una **Legacy API Key**;
4. scaricare `kaggle.json`.

Il file contiene credenziali private.

**Non deve mai essere caricato su GitHub.**

Su Colab il notebook `colab_bootstrap.ipynb` gestisce il caricamento/configurazione del file.

---

# Gestione efficiente del dataset su Colab

Durante lo sviluppo è stato verificato che leggere o copiare decine di migliaia di piccoli file direttamente dal Google Drive montato è estremamente lento.

Il collo di bottiglia principale è l'I/O.

Per questo motivo il dataset non viene utilizzato direttamente dal Drive durante il training.

La strategia scelta è:

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

---

# Archivio persistente SALICON

Su Google Drive viene conservato:

```text
/content/drive/MyDrive/nndl-saliency/salicon.tar
```

L'archivio contiene la struttura corretta:

```text
images/
├── train/
├── val/
└── test/

maps/
├── train/
└── val/

fixations/
├── train/
├── val/
└── test/
```

Durante ogni nuova sessione Colab il file viene estratto in:

```text
/content/data_local
```

---

# Prima preparazione del dataset

La prima volta viene eseguito:

```text
Kaggle
   ↓
download SALICON
   ↓
/content/data_local
   ↓
audit dataset
   ↓
creazione salicon.tar
   ↓
Google Drive
```

Una volta creato:

```text
salicon.tar
```

non è più necessario scaricare SALICON da Kaggle ad ogni nuova sessione.

---

# Sessioni Colab successive

Ad ogni nuova sessione Colab il flusso consigliato è:

```text
1. Attivare GPU
2. Montare Google Drive
3. Clonare / aggiornare repository
4. Installare requirements
5. Eseguire sezione 6bis
6. Eseguire smoke test
7. Training / evaluation
```

La sezione `6bis` di:

```text
notebooks/colab_bootstrap.ipynb
```

estrae automaticamente `salicon.tar` in:

```text
/content/data_local
```

---

# Configurazione dataset

La configurazione principale si trova in:

```text
configs/data.yaml
```

Configurazione attuale:

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

`dataset_root` rappresenta il percorso generico del dataset.

Su Colab viene utilizzata la cache:

```text
/content/data_local
```

Non devono essere inseriti nel codice path personali del tipo:

```text
C:\Users\NomeUtente\...
/Users/nomeutente/...
```

I path condivisi devono passare attraverso le configurazioni del progetto.

---

# Split SALICON

Lo split utilizzato dal progetto è fisso e riproducibile.

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

---

# Manifest dello split

Lo split è stato generato una sola volta e salvato in:

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

Tutti i collaboratori devono utilizzare questo manifest.

**Non bisogna rigenerare individualmente lo split.**

Questo garantisce che:

```text
B0
B1
M1
M1-L
G
M2
```

vengano confrontati esattamente sugli stessi campioni.

---

# Creazione dello split

Lo script utilizzato è:

```text
src/data/splits.py
```

Il risultato verificato è:

```text
Train ufficiale: 10000
Validation ufficiale: 5000

Train: 10000
Tuning: 2500
Internal test: 2500
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

Ogni campione restituisce una struttura simile a:

```python
{
    "image": image_tensor,
    "density_map": density_map_tensor,
    "fixation_path": "...",
    "image_id": "...",
    "split": "..."
}
```

---

# Preprocessing immagini

Il preprocessing viene eseguito direttamente da `SaliconDataset`.

Ogni immagine viene:

1. caricata in RGB;
2. ridimensionata a `256 × 192`;
3. convertita in tensore PyTorch;
4. normalizzata utilizzando le statistiche ImageNet.

Valori utilizzati:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

La shape finale è:

```text
[3, 192, 256]
```

---

# Preprocessing density map

Ogni density map viene:

1. caricata in grayscale;
2. ridimensionata a `256 × 192`;
3. convertita in `float32`;
4. limitata a valori `>= 0`;
5. stabilizzata con epsilon;
6. normalizzata affinché la somma totale sia pari a `1`.

Il valore di epsilon è:

```text
1e-6
```

Formalmente:

```text
P = (P + ε) / Σ(P + ε)
```

La shape finale è:

```text
[1, 192, 256]
```

Questo permette di interpretare la density map come distribuzione spaziale ed è importante per metriche e loss come SIM e KLD.

---

# Data augmentation

Al momento viene utilizzata esclusivamente:

```text
horizontal flip
```

Il flip viene applicato contemporaneamente a:

```text
immagine
+
density map
```

in modo da non distruggere la corrispondenza spaziale.

Non vengono utilizzati:

```text
random crop
rotazioni casuali forti
trasformazioni geometriche aggressive
```

perché altererebbero direttamente la distribuzione spaziale dell'attenzione che il modello deve apprendere.

---

# Fixation data

I fixation file sono disponibili in formato:

```text
.mat
```

Il Dataset attualmente restituisce:

```python
sample["fixation_path"]
```

Esempio:

```text
/content/data_local/fixations/train/COCO_train2014_XXXXXXXXXXXX.mat
```

Il parsing effettivo del contenuto `.mat` verrà implementato nella fase dedicata a NSS/sAUC.

---

# DataLoader

La pipeline è compatibile con:

```python
torch.utils.data.DataLoader
```

È stato verificato un batch di:

```text
4 immagini
+
4 density maps
```

Shape risultanti:

```text
Images:
[4, 3, 192, 256]

Density maps:
[4, 1, 192, 256]
```

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

Lo smoke test controlla automaticamente:

- `train = 10000`;
- `tuning = 2500`;
- `internal_test = 2500`;
- caricamento di campioni dai tre split;
- shape immagini;
- shape density map;
- assenza di NaN;
- assenza di Inf;
- normalizzazione delle density map;
- presenza dei fixation file;
- corretto funzionamento del DataLoader;
- corretto batching.

Il test verifica più campioni per ogni split.

Il risultato attualmente ottenuto è:

```text
============================================================
SMOKE TEST PASSED
============================================================
```

Questo conferma che la pipeline dati è pronta per essere utilizzata dai modelli.

---

# Stato della pipeline dati

```text
SALICON                       ✅
Audit                         ✅
Fixation                      ✅
salicon.tar                   ✅
Cache locale Colab            ✅
Split 10k / 2.5k / 2.5k       ✅
Manifest condiviso            ✅
SaliconDataset                ✅
Preprocessing RGB             ✅
Preprocessing density map     ✅
Horizontal flip sync          ✅
DataLoader                    ✅
Smoke test                    ✅
```

---

# Prossimi passi

## 1. Metriche

Implementare:

```text
CC
SIM
KLD
```

Queste saranno le metriche principali utilizzate per confrontare tutti i modelli.

Dato che le fixation sono disponibili, successivamente implementare anche:

```text
NSS
sAUC
```

---

## 2. B0 — Center Prior

B0 rappresenta la baseline più semplice.

Non richiede una rete neurale.

La predizione viene ottenuta dalla saliency media calcolata sul training set.

Serve a quantificare quanto del problema possa essere spiegato solamente dal **center bias**.

---

## 3. B1 — Baseline neurale

B1 utilizzerà:

```text
ResNet18 pretrained
+
decoder semplice
+
MSE
```

Costituirà la baseline neurale principale.

---

## 4. M1 — Multi-scale

M1 aggiungerà a B1 feature multi-scala provenienti da:

```text
C3
C4
C5
```

tramite skip connections.

La loss rimarrà la stessa di B1:

```text
MSE
```

In questo modo:

```text
B1 → M1
```

isolerà l'effetto dell'architettura multi-scala.

---

## 5. M1-L — Loss

M1-L manterrà esattamente la stessa architettura di M1.

Cambierà solamente la loss:

```text
CC-loss + KLD
```

Quindi:

```text
M1 → M1-L
```

misurerà l'effetto del cambio di funzione obiettivo.

---

## 6. G — Adaptive Center Prior

G rappresenta la componente originale principale del progetto.

Verrà aggiunto un **Adaptive Center Prior** sopra M1-L.

L'idea è imparare un coefficiente:

```text
α(x)
```

dipendente dall'immagine.

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

L'obiettivo è permettere alla rete di imparare quando il center bias è utile e quando invece deve essere ignorato.

G è una parte centrale del progetto e non deve essere eliminato in caso di mancanza di tempo.

---

## 7. M2 — Advanced

M2 utilizzerà un encoder Transformer gerarchico leggero.

La scelta consigliata è utilizzare tramite `timm` un backbone capace di produrre feature multi-scala, ad esempio:

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

# Metriche finali

Metriche principali:

```text
CC
SIM
KLD
```

Metriche aggiuntive grazie alle fixation:

```text
NSS
sAUC
```

Tutti i modelli devono utilizzare:

```text
lo stesso preprocessing
lo stesso split
lo stesso internal test
lo stesso protocollo di evaluation
```

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

Tutti i collaboratori devono lavorare sullo stesso repository.

Prima di iniziare una sessione di sviluppo:

```text
Pull / git pull
```

poi:

```text
sviluppo
↓
test
↓
commit
↓
push
```

È consigliato evitare di modificare contemporaneamente gli stessi file.

Con l'aumentare del codice verranno utilizzati branch separati per feature differenti.

---

# Convenzioni

## Split

Lo split ufficiale è:

```text
results/split_manifest.csv
```

Non deve essere rigenerato dai singoli collaboratori.

---

## Dataset

Il dataset non deve essere caricato su GitHub.

Ogni collaboratore prepara il proprio ambiente utilizzando gli script e il notebook disponibili.

---

## Configurazioni

Le configurazioni condivise devono stare in:

```text
configs/
```

Gli script non devono contenere iperparametri importanti hardcoded quando questi possono essere definiti in configurazione.

---

## Path

Non utilizzare path personali come:

```text
C:\Users\nome\...
/Users/nome/...
```

I path condivisi devono essere configurabili.

Il path:

```text
/content/data_local
```

è specifico della VM Colab ed è definito come cache Colab nel file di configurazione.

---

# File che NON devono essere caricati su GitHub

Non committare:

```text
dataset SALICON
*.jpg
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

I checkpoint dei modelli verranno salvati su:

```text
Google Drive
```

e non nel repository.

Questo permette di mantenere i training persistenti anche quando una sessione Colab termina.

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
checkpoint
stesso protocollo di evaluation
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

---

## Installare dipendenze

```bash
pip install -r requirements-colab.txt
```

---

## Audit dataset

```bash
python scripts/audit_dataset.py --data_dir /content/data_local
```

---

## Smoke test

```bash
python scripts/smoke_test.py
```

Output corretto:

```text
SMOKE TEST PASSED
```

---

# Entry point futuri

Nelle prossime fasi verranno aggiunti almeno:

```text
scripts/train.py
scripts/evaluate.py
```

L'obiettivo finale è poter utilizzare il progetto principalmente con tre operazioni:

```text
smoke test
training
evaluation
```

Non è necessario concentrare tutta la logica in un unico `main.py`.

Un eventuale `main.py` generale o una CLI potranno essere aggiunti alla fine, quando dataset, modelli, training ed evaluation saranno già stabili.

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
✅ Fixation
✅ salicon.tar
✅ Cache locale
✅ Split
✅ Manifest
✅ Dataset PyTorch
✅ Preprocessing
✅ Data augmentation
✅ DataLoader
✅ Smoke test

FASE 3 — METRICHE
⏳ CC
⏳ SIM
⏳ KLD
⏳ NSS
⏳ sAUC

FASE 4 — BASELINE
⏳ B0 Center Prior
⏳ B1 ResNet18 + decoder

FASE 5 — MODELLI
⏳ M1
⏳ M1-L
⏳ G
⏳ M2

FASE 6 — TRAINING / EVALUATION
⏳ Training loop
⏳ Checkpoint/resume
⏳ Evaluation
⏳ Bootstrap CI
⏳ Figure qualitative

FASE 7 — CONSEGNA
⏳ README definitivo
⏳ Riproducibilità da clone pulito
⏳ Report finale
```

---

# Prossimo obiettivo operativo

Il prossimo blocco di sviluppo è:

```text
Implementazione e verifica di:

CC
SIM
KLD
```

Una volta validate le metriche si procederà con:

```text
B0 — Center Prior
```

e successivamente:

```text
B1 — ResNet18 + decoder + MSE
```

Da quel momento inizierà la parte di training e confronto sperimentale dei modelli.