# Contrastive_Learning_Sigmoid_Loss

Questo repository contiene il codice per l'**addestramento e la valutazione di un modello basato su apprendimento contrastivo**, applicato a dati radiologici **reali** (MIMIC-CXR) e **sintetici**.

In entrambe le cartelle principali, gli script per **addestrare, testare e valutare** il modello sono numerati progressivamente: questa numerazione rappresenta l’**ordine consigliato di esecuzione**. Gli script **non numerati** sono utilizzati per la generazione dei dataset, la preparazione dei dati e la definizione dell’architettura del modello Siamese.

> 🔁 Nella cartella [MIMIC](https://github.com/DusoGiulio/Contrastive_Learning_Sigmoid_Loss/tree/main/MIMIC) sono presenti **due script con numero 0**.  
> Il file [`0_Test_Base.py`](https://github.com/DusoGiulio/Contrastive_Learning_Sigmoid_Loss/blob/main/MIMIC/0_Test_Base.py) utilizza una rete Siamese **senza livelli MLP**, con lo scopo di fornire una **baseline comparativa**.  
> In questo caso, la sequenza consigliata di esecuzione sarà: `0 -> 2 -> 3 -> 4`, ricordandosi di **aggiornare correttamente i path** nei vari script.

---

## 📁 Struttura del repository

- 🔬 [**MIMIC**](https://github.com/DusoGiulio/Contrastive_Learning_Sigmoid_Loss/tree/main/MIMIC): contiene i codici relativi ai dati reali del dataset **MIMIC-CXR**.
- 🧪 [**Syntetic**](https://github.com/DusoGiulio/Contrastive_Learning_Sigmoid_Loss/tree/main/Syntetic): contiene i codici per l’elaborazione dei **dati generati sinteticamente**.

---

## ⚠️ Requisiti

I codici sono stati eseguiti con **PyTorch 2.4.0**.

Funzionano **solo se i dati sono scaricati dalla seguente cartella Google Drive**, rispettando la struttura prevista:

📂 [Cartella dei dati – Google Drive](https://drive.google.com/drive/folders/1knHZOF-oiEDl5fsJvqHVz5ClMTquBAbN?usp=sharing)

> ⚠️ Assicurati di scaricare i dati e posizionarli localmente **mantenendo l’organizzazione** prevista.  
> È necessario anche **aggiornare i path nei vari script** per puntare correttamente alle cartelle locali.

### 📦 Contenuto della cartella Google Drive

```plaintext
📁 Cartella principale
│
├── 📁 Result_Syntetic_COSINE
├── 📁 Result_Syntetic_01
├── 📁 Result_MIMIC_COSINE
├── 📁 Result_MIMIC_BASE
├── 📁 Result_MIMIC_01
│
├── 📁 Matrici di similarità usate per addestramento
├── 📁 chexnet
├── 📁 bert
│
├── 📦 Syntetic_dataset.zip
└── 📦 mimic_data.zip
```

- **`Syntetic_dataset.zip`** e **`mimic_data.zip`**: contengono i dataset sintetici e reali usati per l’addestramento e la valutazione.
- **`Matrici di similarità usate per addestramento/`**: include le matrici di similarità (coseno o binarie) utilizzate per guidare l'apprendimento contrastivo.
- **`chexnet/`**: contiene l’encoder visivo pre-addestrato (**DenseNet121**).
- **`bert/`**: contiene l’encoder testuale pre-addestrato (**RadBERT**).
- **`Result_*/`**: ciascuna cartella include:
  - metriche e risultati ottenuti in fase di addestramento/test,
  - file `.pth` con i pesi del modello addestrato, pronti per il caricamento in fase di inferenza.

---

## 📌 Note aggiuntive

- Le cartelle con **"COSINE"** si riferiscono ad addestramenti basati su **matrici di similarità coseno** tra gli embedding.
- Le cartelle con **"01"** indicano l’uso di **matrici di similarità binarie** (0 o 1) per guidare l’ottimizzazione contrastiva.

---
