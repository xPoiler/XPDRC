Se ti piace l'app, puoi sostenermi su https://ko-fi.com/xpoiler ogni donazione è molto apprezzata, XPDRC rimarrà gratuito per sempre!

# XPDRC 1.0 Web DSP

Benvenuti in XPDRC, questa applicazione è uno strumento avanzato di generazione di filtri per l'elaborazione del segnale digitale con un'interfaccia web che comunica direttamente con l'API di Room EQ Wizard per estrarre le risposte all'impulso grezze. Elaborando le risposte all'impulso native generate da REW senza conversioni intermedie di testo o forma d'onda, XPDRC genera filtri FIR a fase minima a latenza zero, o filtri a fase lineare/mista ottimizzati per la correzione acustica ambientale.

La filosofia principale di XPDRC è fornire la massima fedeltà di correzione e aderenza alla fase evitando meticolosamente la generazione di artefatti come pre-ringing, ringing da inversione o rimozione innaturale del campo ambientale.

Cosa può fare XPDRC:

- Elaborazione fino a 8 canali in 9 diverse posizioni.
- Crossover a fase lineare o fase minima di vari tipi e pendenze.
- Correzione di magnitudo sotto la frequenza di Schroeder per mitigare i modi ambientali su tutti i canali separatamente.
- EQ di magnitudo per subwoofer.
- Correzione di magnitudo quasi-anecoica per i diffusori principali sopra la frequenza di Schroeder per il bilanciamento timbrico senza rendere il suono cupo o smorto.
- Inversione della fase in eccesso per i diffusori principali con algoritmi di prevenzione degli artefatti da ringing (linearizzazione della fase).
- EQ del ritardo di gruppo per il sub con analisi iterativa della risonanza per prevenire artefatti di pre e post ringing.
- Generazione completamente automatica di Virtual Bass Array per minimizzare i modi ambientali (solo se è presente il canale LFE).
- Bilanciamento automatico del volume di tutti i canali.
- Allineamento temporale automatico o manuale di tutti i canali.
- Allineamento automatico o manuale di un subwoofer con i diffusori principali (singolo subwoofer, per ora).
- Generazione di filtri da 4096 tap fino a 131072 per la massima precisione.

e altro ancora... le impostazioni e le preferenze possono essere facilmente cambiate dall'utente in ogni momento tramite l'interfaccia web locale disponibile su localhost:5000; quasi ogni singola variabile nello script può essere modificata dalla UI. Qualora l'utente lo desideri, consigliamo vivamente le impostazioni predefinite per facilità d'uso e migliori risultati nella maggior parte delle situazioni.

## Installazione e Configurazione

Far funzionare XPDRC è progettato per essere il più semplice possibile.

1. **Installa Python**: Assicurati di avere Python 3.10+ installato sul tuo sistema.
2. **Clona/Scarica**: Scarica il codice sorgente sul tuo computer locale.
3. **Avvia REW**: Assicurati che Room EQ Wizard (REW) sia aperto e che il suo server API sia abilitato (porta predefinita 4735).
4. **Lancia**: Fai doppio clic sul file **`Run XPDRC.bat`** nella directory del progetto.

Lo script batch eseguirà automaticamente:
- Creazione di un ambiente virtuale (`.venv`) per mantenere il sistema pulito.
- Installazione di tutte le dipendenze necessarie.
- Avvio dell'applicazione e apertura dell'interfaccia nel browser predefinito.

## Come usare

Una volta avviata, l'applicazione si aprirà automaticamente nel browser all'indirizzo `http://127.0.0.1:5000`.

### Configurazione Guidata
Il modo più semplice per iniziare è cliccare sul pulsante **"Lancia l'Assistant de Configuration"**. Questo assistente ti guiderà attraverso:
- L'esecuzione delle misurazioni richieste in REW.
- L'identificazione automatica di diffusori e subwoofer.
- La configurazione dei parametri di elaborazione di base.

### Configurazione Manuale
Se preferisci il controllo manuale, puoi:
- Inserire direttamente gli ID delle misurazioni REW.
- Regolare le frequenze di crossover, i boost della curva house e i limiti di frequenza.
- Attivare o disattivare funzionalità avanzate e cambiarne i parametri operativi.

Una volta configurato, clicca su **"Genera FIR Base"** (Fase 1) seguito dal pulsante di generazione verde in Fase 2. L'app produrrà file di risposta all'impulso `.wav` per ogni diffusore elaborato e un file di EQ globale per il tuo convolutore. Applica un guadagno negativo globale nel tuo sistema per prevenire il clipping.

## Documentazione

Per un'analisi completa di come funzionano matematicamente gli algoritmi di elaborazione del segnale digitale, consulta la pagina di Documentazione accessibile direttamente dall'interfaccia web. La documentazione dettaglia la pipeline di elaborazione predefinita, inclusi il frequency dependent windowing, la linearizzazione della fase in eccesso e l'analisi iterativa della risonanza, oltre a funzionalità avanzate attivabili come il design a Fase Mista e l'inversione spettrale regolarizzata di Kirkeby.

## Disclaimer Beta e Supporto

Si prega di notare che XPDRC versione 1.0 è attualmente in Beta. Sebbene le impostazioni predefinite principali della pipeline di generazione DSP siano state pesantemente testate, alcune combinazioni di funzionalità potrebbero non funzionare ancora perfettamente in tutti gli ambienti informatici.

Se riscontri anomalie o bug, o se hai semplicemente domande sul funzionamento degli algoritmi, non esitare a contattarmi. Puoi scrivermi direttamente a xpoileremmo@gmail.com.
