# PolyMM-Pro v5 📈🧬
**Institutional-Grade High-Frequency Market Maker for Polymarket**

Benvenuto in PolyMM-Pro. Questo sistema è stato progettato con architettura quantitativa asincrona (Tier-1) per fornire liquidità bilaterale nei mercati predittivi di Polymarket, catturando lo spread e gestendo algoritmicamente l'esposizione al rischio (Inventory Risk e Adverse Selection).

---

## 🛠️ Architettura e Moduli
- **Sensore VWAP (L2 Orderbook)**: Non usa il Mid-Price semplice, ma calcola il *Volume-Weighted Average Price* guardando la profondità del book.
- **Micro-Price Volatility**: Allarga o stringe dinamicamente il *Base Spread* in base alla deviazione standard del mercato negli ultimi tick.
- **Exponential Inventory Skew**: Più token accumuli su un lato (YES o NO), più il bot saboterà i prezzi in quella direzione per disincentivare acquisti ulteriori e forzare lo scarico dell'inventario.
- **Kelly Fractional Sizing**: Riduce quadraticamente la dimensione della Size scambiabile (Es. 100 USDC -> 10 USDC -> 0 USDC) man mano che ti avvicini al rischio di saturazione del portafoglio.
- **Queue Priority (Smart Routing)**: Non spamma l'invio e cancellazione di ordini. Attende una *Tolleranza* (es. 0.5%) per mantenere il privilegio del primo posto nella coda del Matching Engine di Polymarket.
- **Crash Recovery**: Ad ogni operazione esegue il flush dello stato (PnL, Inventario, Ordini Aperti) in un bunker `data/state.json`. Se l'elettricità salta, non perde la memoria.
- **Telemetry Server**: Espone una porta locale `http://localhost:8192/metrics` per farti monitorare i log di rischio da applicazioni esterne (es. Grafana).

---

## 🚀 Manuale Operativo: Come andare in LIVE

Poiché l'architettura è ormai definita (Fase 11 conclusa), devi smettere i panni dell'ingegnere e indossare quelli dell'operatore finanziario. Ecco i passi esatti:

### Passo 1: Configurazione Chiavi e Denaro
1. Entra nel [sito di Polymarket](https://polymarket.com/) accedendo con tuo Wallet. 
2. Assicurati di avere fondi su Rete **Polygon**:
   - Qualche **MATIC/POL** per pagare il Gas dei ritiri.
   - I tuoi **USDC.e** per l'esposizione del bot.
3. Clicca sul Profilo (in alto a destra) -> Settings -> **Developer Keys**. Crea una chiave API L2 per il CLOB (Orderbook).
4. Copia la tua **Private Key (PK)** e incollala dentro `polymm-pro/.env` (Il file che abbiamo appena creato, non pusharlo MAI online per sicurezza).

### Passo 2: Profilazione del Rischio (`config.yaml`)
Apri il file `config.yaml` dal tuo editor:
- Scegli la **Liquidità di Fuoco** (`quote_size`): Quanti USDC vuoi quotare su ogni lato? Metti una cifra che puoi permetterti (Es. `10.0`).
- Scegli lo **Spread Iniziale** (`base_spread`): Quanti centesimi di profitto pretendi dal rumore di mercato? (Es. `0.04` = 4%).
- Imposta i **Limiti di Sicurezza** (`max_inventory`): Quanti token YES/NO il bot può tenere in pancia prima di spegnersi? (Es. `100`).
- Identifica i **Token IDs** del Mercato in cui vuoi operare (estrai l'Asset ID finale dall'URL del mercato Polymarket) e inseriscili nella lista `tokens`.
- Accendi l'Interruttore finale: Cambia `live_trading_enabled: false` a **`live_trading_enabled: true`**.

### Passo 3: Deploy asincrono (24/7)
Il Market Making richiede che il bot acceda al mercato e quoti in maniera passiva. Se esegui il bot sul tuo PC:
1. **Lancio Base**: Esegui `$env:PYTHONPATH="."; python -m uv run src/main.py`
2. Se spegni il PC, il bot muore e potrebbe nascere un rischio di **Gap Notturno** qualora gli ordini limite sopravvivessero e le news americane uscissero prima che tu riaccenda il PC.

🌟 **Consiglio da Pro (Cloud Server VPS)**:
Noleggia un Virtual Private Server (VPS) da 5€ al mese (es. DigitalOcean "Droplet", Hetzner, AWS EC2 nano):
1. Installa Linux Ubuntu.
2. Clona questo progetto e installa le dipendenze con `uv`.
3. Inserisci il `.env`.
4. Avvia il bot sfruttando **Screens** o **SystemD** in backgrond: `nohup python -m uv run src/main.py &`
In questo modo lui quoterà in un data-center super protetto, con latenza <10ms verso i server Clob di Polymarket, operando per te anche quando dormi.

---
*Happy Trading, PolyMM-Pro Operator.* 📈⚡
