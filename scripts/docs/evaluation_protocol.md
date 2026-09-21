# Evaluation Protocol

## Metriche principali

Il protocollo di valutazione principale utilizza:

- CC (Correlation Coefficient)
- SIM (Similarity)
- KLD (Kullback-Leibler Divergence)

Queste tre metriche devono essere riportate per tutti i modelli
confrontati nel percorso sperimentale principale.

## Metriche fixation-based

NSS e sAUC restano implementate nel progetto ma non fanno parte
del percorso critico della valutazione principale.

- NSS: opzionale, da includere se il tempo disponibile lo consente.
- sAUC: opzionale, salvo richiesta esplicita del corso o decisione
  successiva del team.

Le implementazioni esistenti di NSS e sAUC non devono essere rimosse.

## Split di valutazione

Durante sviluppo, debugging e selezione dei modelli viene utilizzato
esclusivamente lo split `tuning`.

Lo split `internal_test` resta congelato fino al freeze definitivo
di architetture, loss e protocollo sperimentale.

L'accesso a `internal_test` richiede una valutazione finale esplicita.

## Regola di confronto

I confronti per-image tra modelli devono essere effettuati tramite
`image_id`.

Non è consentito assumere che due CSV siano allineati semplicemente
in base all'ordine delle righe.

ID mancanti o duplicati devono essere segnalati come errore.