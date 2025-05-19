# NLP-Wikipedia-Agent-Mistral

Ce projet est un agent qui utilise le modèle de langage Mistral-7B pour répondre à des questions en s'appuyant sur des informations extraites de Wikipédia.

## Fonctionnalités

* **Recherche d'informations sur Wikipédia :** L'agent peut rechercher des informations pertinentes sur un sujet donné en utilisant la bibliothèque `WikipediaLoader` de Langchain.
* **Utilisation de Mistral-7B :** Le modèle Mistral-7B est utilisé pour générer des réponses aux questions posées.
* **Récupération Augmentée Génération (RAG) :** L'agent utilise une approche RAG pour combiner les informations extraites de Wikipédia avec la capacité de génération de Mistral-7B.
* **Base de connaissances vectorielle :** Les informations de Wikipédia sont stockées dans une base de connaissances vectorielle (Chroma) pour une récupération efficace.
* **Gestion de la mémoire GPU :** Le code inclut une configuration pour charger le modèle Mistral-7B en utilisant la quantification bitsandbytes pour réduire l'utilisation de la mémoire GPU.

## Prérequis

* Python 3.x
* Un compte Hugging Face et une clé API
* Les bibliothèques listées dans `requirements.txt`

## Installation

1.  Clonez ce dépôt :

    ```bash
    git clone https://github.com/Thibaut3/NLP-Wikipedia-Agent-Mistral.git
    cd votre-repo-nom
    ```

2.  Créez un environnement virtuel (recommandé) :

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    venv\Scripts\activate  # Sur Windows
    ```

3.  Installez les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

4.  Configurez votre clé API Hugging Face :

    ```bash
    export HUGGINGFACE_API_KEY="votre_clé_api" # Remplacez "votre_clé_api" par votre clé réelle
    ```
    ou
     ```python
     import os
     os.environ["HUGGINGFACE_API_KEY"] = "YOUR_HUGGINGFACE_API_KEY"
     ```

## Utilisation

1.  Exécutez le script `main.py` :

    ```bash
    python main.py
    ```

    Le script exécutera l'agent avec un exemple de question et de sujet de recherche.

## Configuration

* `huggingface_token`:  Votre jeton d'authentification Hugging Face.  Il est préférable de le définir comme variable d'environnement.
* `bnb_config`: Configuration pour le chargement quantifié du modèle Mistral-7B.  Peut être ajusté pour différents niveaux de performance et d'utilisation de la mémoire.
* `model_id`: L'identifiant du modèle Mistral-7B à utiliser.
* `embeddings`: Le modèle de plongement de phrase à utiliser.
* `template`: Le modèle de prompt pour l'agent. Peut être personnalisé pour modifier le comportement de l'agent.
* `load_max_docs`: Nombre maximum de documents à charger à partir de Wikipedia.
* `chunk_size`: Taille des morceaux de texte à diviser à partir des documents Wikipedia.
* `chunk_overlap`: Chevauchement entre les morceaux de texte.
* `k`: Nombre de documents à récupérer de la base de connaissances vectorielle.

## Exemple de sortie

Question: Quelles sont les applications principales de l'intelligence artificielle dans la vie quotidienne?
Recherche d'informations sur: Intelligence artificielle
Réponse:
L'intelligence artificielle a de nombreuses applications dans la vie quotidienne.  Voici quelques exemples :
 - Assistants virtuels: Siri, Google Assistant et Alexa peuvent répondre à des questions, définir des rappels et contrôler des appareils domestiques.
 - Recommandations: Les algorithmes de recommandation sont utilisés par Netflix, Amazon et YouTube pour suggérer du contenu susceptible de vous intéresser.
 - Filtres anti-spam: L'IA est utilisée pour identifier et filtrer les courriels indésirables.
 - Détection de fraude: Les systèmes d'IA peuvent analyser les transactions financières pour détecter les activités frauduleuses.
 - Voitures autonomes: L'IA est essentielle au développement des véhicules autonomes.
 - Diagnostic médical: L'IA peut aider les médecins à diagnostiquer des maladies à partir d'images médicales.
 - Traduction automatique: Les outils de traduction automatique comme Google Traduction utilisent l'IA pour traduire du texte et de la parole.

## Dépendances

* langchain
* langchain\_community
* torch
* transformers
* accelerate
* bitsandbytes
* chromadb
* wikipedia
* sentence-transformers

## Notes

* Assurez-vous d'avoir suffisamment de mémoire GPU si vous exécutez le modèle sans quantification.
* La qualité des réponses de l'agent dépend de la qualité des informations disponibles sur Wikipédia.
* Vous pouvez ajuster le modèle de prompt pour modifier le comportement de l'agent.
