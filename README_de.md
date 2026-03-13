Wenn Ihnen die App gefällt, können Sie mich unter https://ko-fi.com/xpoiler unterstützen. Jede Spende wird sehr geschätzt, XPDRC wird für immer kostenlos bleiben!

# XPDRC 1.3 Web DSP

Willkommen bei XPDRC. Diese Anwendung ist ein fortschrittliches Werkzeug zur Generierung von Filtern für die digitale Signalverarbeitung mit einer Weboberfläche, die direkt mit der Room EQ Wizard API kommuniziert, um rohe Impulsantworten zu extrahieren. Durch die Verarbeitung der von REW generierten nativen Impulsantworten ohne zwischengeschaltete Text- oder Wellenformkonvertierungen generiert XPDRC latenzfreie Minimalphasen- oder latenzoptimierte Linearphasen-/Mixed-Phase-Filter für die Raumakustikkorrektur.

Die Kernphilosophie von XPDRC besteht darin, eine Korrektur mit höchster Wiedergabetreue und Phasenadhärenz zu bieten, während die Erzeugung von Artefakten wie Pre-Ringing, Inversions-Ringing oder unnatürliche Raumfeldentfernung akribisch vermieden wird.

Was XPDRC leisten kann:

- Verarbeitung von bis zu 8 Kanälen an 9 verschiedenen Positionen.

- Linearphasen- oder Minimalphasen-Frequenzweichen verschiedener Typen und Steilheiten.

- Magnitudenkorrektur unterhalb der Schroeder-Frequenz zur Milderung von Raummoden auf allen Kanälen separat.

- Subwoofer-Magnituden-EQ.

- Quasi-anechoische Magnitudenkorrektur für Hauptlautsprecher oberhalb der Schroeder-Frequenz zur Klangbalancierung, ohne den Klang stumpf zu machen.

- Excess-Phase-Inversion für die Hauptlautsprecher mit Algorithmen zur Vermeidung von Ringing-Artefakten (Phasenlinearisierung).

- Gruppenlaufzeit-EQ für den Subwoofer mit iterativer Ringing-Analyse zur Vermeidung von Pre- und Post-Ringing-Artefakten.

- Vollautomatische Generierung von Virtual Bass Arrays zur Minimierung von Raummoden (nur wenn ein LFE-Kanal vorhanden ist).

- Automatischer Lautstärkeabgleich aller Kanäle.

- Automatische oder manuelle Zeitausrichtung aller Kanäle.

- Automatische oder manuelle Ausrichtung eines Subwoofers auf die Hauptlautsprecher (derzeit einzelner Subwoofer).

- Filtergenerierung von 32768 Taps bis hin zu 131072 für maximale Präzision.

und mehr... Einstellungen und Präferenzen können vom Benutzer jederzeit einfach über die lokale Weboberfläche unter localhost:5000 geändert werden; fast jede einzelne Variable im Skript kann über die UI angepasst werden. Für eine einfache Bedienung und beste Ergebnisse in den meisten Situationen empfehlen wir dringend die Standardeinstellungen.

## Installation & Einrichtung

Die Inbetriebnahme von XPDRC ist so einfach wie möglich gestaltet.

1.  **Python installieren**: Stellen Sie sicher, dass Python 3.10+ auf Ihrem System installiert ist.
2.  **Klonen/Herunterladen**: Laden Sie den Quellcode auf Ihren lokalen Rechner herunter oder laden Sie die Zip-Datei aus dem Releases-Bereich herunter.
3.  **REW ausführen**: Stellen Sie sicher, dass Room EQ Wizard (REW) geöffnet und sein API-Server aktiviert ist (Standardport 4735).

**WICHTIG**: Gehen Sie bei der Ersteinrichtung in REW auf die Registerkarte "Analysis" und ändern Sie "For imports set t=0 at impulse peak" in "For imports set t=0 at first sample". Wenn Sie dies nicht tun, wird das mathematische Ergebnis nicht wie erwartet ausfallen.
 
4.  **Starten**: Doppelrechnen Sie auf die Datei **`Run XPDRC.bat`** im Projektverzeichnis.

Das Batch-Skript wird automatisch:

- Eine virtuelle Umgebung (`.venv`) erstellen, um Ihr System sauber zu halten.
 
- Alle notwendigen Abhängigkeiten installieren (`numpy`, `scipy`, `flask`, etc.).
 
- Die Anwendung starten und die Oberfläche in Ihrem Standardbrowser öffnen.

### Manuelle Installation (Alle Betriebssysteme)

XPDRC kann auf jedem Betriebssystem (Windows, macOS, Linux) über das Terminal ausgeführt werden:

1.  Öffnen Sie ein Terminal im XPDRC-Ordner.
2.  Installieren Sie die Abhängigkeiten:
    ```bash
    pip install -r requirements.txt
    ```
3.  Starten Sie die Anwendung:
    ```bash
    python app.py
    ```

## Anwendung

Sobald die Anwendung läuft, öffnet sie sich automatisch in Ihrem Browser unter `http://127.0.0.1:5000`.

### Geführte Einrichtung

Der einfachste Weg zu beginnen ist ein Klick auf die Schaltfläche **"Konfigurationsassistent starten"**. Dieser Assistent führt Sie durch:

- Das Durchführen der erforderlichen Messungen in REW.
  
- Die automatische Identifizierung von Lautsprechern und Subwoofern.
 
- Das Einrichten der grundlegenden Verarbeitungsparameter.

### Manuelle Konfiguration

Wenn Sie die manuelle Steuerung bevorzugen, können Sie:

- REW-Mess-IDs direkt eingeben.
  
- Trennfrequenzen, House-Curve-Boosts und Frequenzgrenzen anpassen.
  
- Erweiterte Funktionen ein- und ausschalten und deren Betriebsparameter ändern.


**WICHTIG**: Messungen sollten mit einer Zeitreferenz durchgeführt und in REW nicht verändert werden (z. B. manuelles Versetzen), da das Skript sonst Fehler auswirkt oder fehlerhafte Filter generiert. Glättung (Smoothing) wird unterstützt.

Klicken Sie nach der Konfiguration auf **"Basis-FIR generieren"** (Phase 1), gefolgt von der grünen Generierungsschaltfläche in Phase 2. Die App erstellt `.wav`-Impulsantwortdateien für jeden verarbeiteten Lautsprecher und eine globale EQ-Datei für Ihren Convolver. Sie finden diese Dateien im Verzeichnis von XPDRC. Wenden Sie in Ihrem System eine globale negative Verstärkung an, um Clipping zu vermeiden.

## Dokumentation

Für eine umfassende Aufschlüsselung der mathematischen Funktionsweise der digitalen Signalverarbeitungsalgorithmen lesen Sie bitte die Dokumentationsseite, die direkt über die Weboberfläche zugänglich ist. Die Dokumentation beschreibt die Standard-Verarbeitungspipeline, einschließlich frequenzabhängiger Fensterung, Phasenlinearisierung und iterativer Ringing-Analyse sowie erweiterte umschaltbare Funktionen wie Mixed-Phase-Design und Kirkeby-regularisierte Spektralinversion.

## Beta-Haftungsausschluss und Support

Bitte beachten Sie, dass sich XPDRC Version 1.2 derzeit in der Beta-Phase befindet. Obwohl die wichtigsten Standardeinstellungen der DSP-Generierungspipeline intensiv getestet wurden, funktionieren einige Funktionskombinationen möglicherweise noch nicht in allen Computerumgebungen perfekt.

Wenn Sie auf Anomalien oder Fehler stoßen oder einfach Fragen zur Funktionsweise der Algorithmen haben, können Sie sich gerne an mich wenden. Sie erreichen mich direkt unter xpoileremmo@gmail.com.
