Si vous aimez l'application, vous pouvez me soutenir sur https://ko-fi.com/xpoiler. Chaque don est très apprécié, XPDRC restera gratuit pour toujours !

# XPDRC 1.0 Web DSP

Bienvenue sur XPDRC. Cette application est un outil avancé de génération de filtres de traitement numérique du signal avec une interface web qui communique directement avec l'API de Room EQ Wizard pour extraire les réponses impulsionnelles brutes. En traitant les réponses impulsionnelles natives générées par REW sans conversions textuelles ou de formes d'onde intermédiaires, XPDRC génère des filtres FIR à phase minimale sans latence, ou des filtres à phase linéaire/mixte optimisés pour la latence pour la correction acoustique des pièces.

La philosophie fondamentale d'XPDRC est de fournir la correction et l'adhérence de phase les plus fidèles tout en évitant méticuleusement la génération d'artéfacts tels que le pré-écho, le bourdonnement d'inversion ou la suppression non naturelle du champ de la pièce.

Ce que XPDRC peut faire :

- Traitement de jusqu'à 8 canaux dans 9 positions différentes.

- Filtres de croisement (crossover) à phase linéaire ou minimale de divers types et pentes.

- Correction de magnitude en dessous de la fréquence de Schroeder pour atténuer les modes de pièce sur tous les canaux séparément.

- Égalisation de magnitude du caisson de basses (subwoofer).

- Correction de magnitude quasi-anéchoïque pour les enceintes principales au-dessus de Schroeder pour l'équilibrage tonal sans ternir le son.

- Inversion de phase excessive pour les enceintes principales avec des algorithmes de prévention des artéfacts de bourdonnement (linéarisation de phase).

- Égalisation du temps de propagation de groupe pour le caisson avec analyse itérative de la résonance pour prévenir les artéfacts de pré et post-écho.

- Génération complètement automatique de Virtual Bass Array pour minimiser les modes de pièce (uniquement si un canal LFE est présent).

- Équilibrage automatique du volume de tous les canaux.

- Alignement temporel automatique ou manuel de tous les canaux.

- Alignement automatique ou manuel d'un caisson de basses avec les enceintes principales (un seul caisson, pour l'instant).

- Génération de filtres de 32768 taps jusqu'à 131072 pour une précision maximale.

et plus encore... les réglages et préférences peuvent être facilement modifiés par l'utilisateur à tout moment via l'interface web locale disponible sur localhost:5000 ; presque chaque variable du script peut être modifiée depuis l'interface utilisateur. Pour une facilité d'utilisation et de meilleurs résultats dans la majorité des situations, nous recommandons vivement les réglages par défaut.

## Installation et Configuration

La mise en route de XPDRC est conçue pour être aussi simple que possible.

1.  **Installer Python** : Assurez-vous d'avoir Python 3.10+ installé sur votre système.
2.  **Cloner/Télécharger** : Téléchargez le code source sur votre machine locale ou téléchargez le fichier zip depuis la section des versions.
3.  **Lancer REW** : Assurez-vous que Room EQ Wizard (REW) est ouvert et que son serveur API est activé (port par défaut 4735).

**IMPORTANT** : Pour la première configuration, allez dans l'onglet "Analysis" de REW et remplacez "For imports set t=0 at impulse peak" par "For imports set t=0 at first sample". Si vous ne le faites pas, les calculs ne donneront pas le résultat attendu.
 
4.  **Lancer** : Double-cliquez sur le fichier **`Run XPDRC.bat`** dans le répertoire du projet.

Le script batch effectuera automatiquement :

- La création d'un environnement virtuel (`.venv`) pour garder votre système propre.
 
- L'installation de toutes les dépendances nécessaires (`numpy`, `scipy`, `flask`, etc.).
 
- Le démarrage de l'application et l'ouverture de l'interface dans votre navigateur par défaut.

### Installation Manuelle (Tous les systèmes d'exploitation)

XPDRC peut être lancé sur n'importe quel système d'exploitation (Windows, macOS, Linux) via le terminal :

1.  Ouvrez un terminal dans le dossier XPDRC.
2.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
3.  Lancez l'application :
    ```bash
    python app.py
    ```

## Comment utiliser

Une fois l'application lancée, elle s'ouvrira automatiquement dans votre navigateur à l'adresse `http://127.0.0.1:5000`.

### Configuration Guidée

Le moyen le plus simple de commencer est de cliquer sur le bouton **"Lancer l'Assistant de Configuration"**. Cet assistant vous guidera pour :

- Effectuer les mesures requises dans REW.
  
- Identifier automatiquement les enceintes et les caissons de basses.
 
- Configurer les paramètres de traitement de base.

### Configuration Manuelle

Si vous préférez un contrôle manuel, vous pouvez :

- Saisir directement les identifiants de mesure REW.
  
- Ajuster les fréquences de coupure, les boosts de courbe cible (house curve) et les limites de fréquence.
  
- Activer ou désactiver les fonctionnalités avancées et modifier leurs paramètres de fonctionnement.


**IMPORTANT** : Aucune égalisation (smoothing) ne doit être appliquée aux mesures. Les mesures doivent être effectuées avec une référence temporelle et ne pas être modifiées dans REW, sinon le script affichera des erreurs ou générera des filtres incorrects.

Une fois configuré, cliquez sur **"Générer FIR de Base"** (Phase 1) suivi du bouton de génération vert dans la Phase 2. L'application produira des fichiers de réponse impulsionnelle `.wav` pour chaque enceinte traitée et un fichier d'égalisation globale pour votre convolveur. Vous pouvez trouver ces fichiers dans le répertoire de XPDRC. Appliquez un gain négatif global dans votre système pour éviter l'écrêtage.

## Documentation

Pour une analyse complète du fonctionnement mathématique des algorithmes de traitement numérique du signal, veuillez vous référer à la page Documentation accessible directement via l'interface web. La documentation détaille le pipeline de traitement par défaut, y compris le fenêtrage dépendant de la fréquence, la linéarisation de la phase en excès et l'analyse itérative de la résonance, ainsi que les fonctionnalités avancées activables telles que la conception à phase mixte et l'inversion spectrale régularisée de Kirkeby.

## Clause de non-responsabilité Beta et Support

Veuillez noter que la version 1.0 d'XPDRC est actuellement en version Beta. Bien que les principaux réglages par défaut du pipeline de génération DSP aient été testés intensivement, certaines combinaisons de fonctionnalités pourraient ne pas encore fonctionner parfaitement dans tous les environnements informatiques.

Si vous rencontrez des anomalies ou des bogues, ou si vous avez simplement des questions sur le fonctionnement des algorithmes, n'hésitez pas à me contacter. Vous pouvez me contacter directement à xpoileremmo@gmail.com.
