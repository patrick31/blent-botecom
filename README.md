# Assistant de service client e-commerce

Projet Python de formation visant à construire un assistant conversationnel capable de répondre aux questions libres d'un client authentifié sur ses commandes e-commerce.

L'assistant utilise un LLM Hugging Face via LangChain pour analyser la demande utilisateur et générer la réponse finale. La base SQLite fournie dans `infos/orders.db` reste la source de vérité, et les accès aux commandes sont protégés côté code par des requêtes SQL paramétrées filtrées sur l'utilisateur authentifié.

## Note Sur Les Environnements Mac Et CUDA

Le code a été réalisé et testé sur macOS par confort de développement. Pour garder le même code applicatif sur Mac et sur la plateforme cible, le lanceur `run.sh` propose deux modes :

- `--cuda` : mode à utiliser pour installer, tester et lancer le projet sur la plateforme cible demandée, équipée de GPU Nvidia/CUDA;
- `--mac` : mode de spécial pour tester localement sur un Mac Apple Silicon disposant d'une mémoire unifiée suffisante.

Pour la plateforme cible du projet, l'installation et le lancement doivent donc se faire avec `--cuda`. 

Le mode `--mac` utilise le modèle MLX quantifié `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`; prévoir au minimum `32768 Mo` de mémoire unifiée, avec `64 Go` recommandés pour exécuter le modèle plus confortablement.

## Installation Rapide

### Cloner le projet
```bash
git clone https://github.com/patrick31/blent-botecom.git
cd blent-botecom
```

### Créer un environnement virtuel depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

### Pour MAC :

Installer les dépendances Mac Apple Silicon :

```bash
pip install -e ".[dev,ui,mac]"
```

ou avec `run.sh` :

```bash
./run.sh --mac --install
```


Vérifier l'installation Mac sans charger le modèle avec `run.sh` :

```bash
./run.sh --mac --check
```

`./run.sh --mac` exécute aussi cette vérification automatiquement avant un lancement CLI ou Streamlit. L'option `--check` sert uniquement à diagnostiquer sans lancer l'application.

### Pour CUDA

Installer les dépendances Nvidia/CUDA :

```bash
pip install -e ".[dev,ui,cuda]"

```

ou avec `run.sh` :

```bash
./run.sh --cuda --install
```

Vérifier l'installation CUDA sans charger le modèle avec `run.sh` :

```bash
./run.sh --cuda --check
```

`./run.sh --cuda` exécute aussi cette vérification automatiquement avant un lancement CLI ou Streamlit. L'option `--check` sert uniquement à diagnostiquer sans lancer l'application.


## Configuration LLM

Le code applicatif reste identique quel que soit le matériel. Le choix du backend se fait avec des variables d'environnement.

### Mac Apple Silicon

Configurer le backend MLX :

```bash
export LLM_BACKEND="mlx"
export HF_LLM_MODEL_ID="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
export LLM_TEMPERATURE="0.0"
export MAX_NEW_TOKENS="180"
```

Tip avec `run.sh` : `./run.sh --mac ...` configure ces variables automatiquement.

Tests pytest avec `run.sh` :

```bash
./run.sh --mac --test
```

### Machine Nvidia/CUDA

La description du projet indique une sandbox avec `2 GPU Nvidia 24 Gio`, soit 48 Gio de VRAM cumulée.

Configurer le backend Transformers :

```bash
export LLM_BACKEND="transformers"
export HF_LLM_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
export HF_DEVICE_MAP="auto"
export HF_TORCH_DTYPE="auto"
export HF_LOAD_IN_4BIT="true"
export LLM_TEMPERATURE="0.0"
export MAX_NEW_TOKENS="180"
```

Tip avec `run.sh` : `./run.sh --cuda ...` configure ces variables automatiquement.

Tests pytest avec `run.sh` :

```bash
./run.sh --cuda --test
```

## Lancement CLI

La CLI peut être lancée avec un email explicite ou sans email. Si aucun email n'est fourni, elle affiche un menu numéroté avec les utilisateurs présents dans la base de test.

Question unique avec email explicite :

```bash
PYTHONPATH=src python3 -m customer_service_bot.cli \
  --email "pellentesque.ultricies@protonmail.com" \
  --question "Ou en est ma commande 4 ?"
```

Question unique avec sélection de l'utilisateur au démarrage :

```bash
PYTHONPATH=src python3 -m customer_service_bot.cli \
  --question "Ou en est ma commande 4 ?"
```

Tip avec `run.sh` sur Mac :

```bash
./run.sh --mac \
  --question "Ou en est ma commande 4 ?"
```

Tip avec `run.sh` sur Nvidia/CUDA :

```bash
./run.sh --cuda \
  --question "Ou en est ma commande 4 ?"
```

Mode interactif avec sélection de l'utilisateur au démarrage :

```bash
./run.sh --mac
```

Mode interactif avec email explicite :

```bash
./run.sh --mac \
  --email "pellentesque.ultricies@protonmail.com"
```

Dans ce mode, la CLI demande les questions une par une jusqu'à `exit`.

Tester le contrôle d'accès avec une commande appartenant à un autre client :

```bash
PYTHONPATH=src python3 -m customer_service_bot.cli \
  --email "pellentesque.ultricies@protonmail.com" \
  --question "Ou en est ma commande 1 ?"
```

Tip avec `run.sh` :

```bash
./run.sh --mac \
  --email "pellentesque.ultricies@protonmail.com" \
  --question "Ou en est ma commande 1 ?"
```

## Lancement Streamlit

Après avoir configuré le backend, lancer l'interface web :

```bash
PYTHONPATH=src streamlit run app_streamlit.py
```

Tip avec `run.sh` sur Mac :

```bash
./run.sh --mac --streamlit
```

Tip avec `run.sh` sur Nvidia/CUDA :

```bash
./run.sh --cuda --streamlit
```

## Objectifs Pédagogiques

- Utiliser Python pour connecter un assistant à une base SQL.
- Utiliser LangChain avec un LLM Hugging Face.
- Analyser des questions client formulées librement.
- Générer une réponse naturelle, professionnelle et compréhensible.
- Transformer les statuts techniques (`invoiced`, `shipped`, `delivered`) en langage client.
- Protéger les données d'un utilisateur authentifié contre les accès croisés.
- Mettre en place un routage sémantique par LLM pour refuser les demandes hors périmètre.

## Données

Le dossier `infos/` contient :

- `orders.db` : base SQLite utilisée par l'application.
- `orders.csv` : export des commandes.
- `users.csv` : export des utilisateurs.
- `description.md` et `etapes.md` : consignes du projet.

Tables principales :

- `users` : informations client (`user_id`, prénom, nom, email, ville, etc.).
- `orders` : commandes (`order_id`, `user_id`, `status`, dates d'achat, d'expédition et de livraison).

## Architecture

```text
.
├── app_streamlit.py
├── infos/
├── pyproject.toml
├── README.md
├── run.sh
├── src/
│   └── customer_service_bot/
│       ├── assistant.py
│       ├── cli.py
│       ├── config.py
│       ├── database.py
│       ├── llm.py
│       └── router.py
└── tests/
    └── test_assistant.py
```

Rôles des modules :

- `database.py` : accès SQLite et filtrage obligatoire par utilisateur authentifié.
- `router.py` : routage sémantique par LLM, extraction de l’intention, validation du périmètre et réparation encadrée des sorties JSON tronquées.
- `llm.py` : fabrique des backends Hugging Face/MLX et génération de la réponse finale en langage naturel.
- `assistant.py` : orchestration de la conversation, appel du routeur LLM, récupération SQL sécurisée et construction des faits métier.
- `cli.py` : interface en ligne de commande.
- `app_streamlit.py` : interface web optionnelle pour démonstration.
- `run.sh` : raccourci pratique pour configurer le backend puis lancer l'application.

## Sécurité Et Limites

Le LLM analyse la question libre et rédige la réponse, mais il ne génère pas de SQL exécutable librement. Les requêtes SQL sont écrites côté application, paramétrées, et incluent toujours le `user_id` de l'utilisateur authentifié.

Mesures mises en place :

- analyse LLM structurée de la demande avant récupération SQL;
- prompt de routage robuste contre les demandes hors périmètre et les tentatives d'injection;
- filtrage des commandes par `user_id` dans `OrderRepository`;
- génération LLM limitée aux faits SQL déjà autorisés;
- message neutre lorsqu'une commande existe peut-être mais n'appartient pas au client.

Limites connues :

- Le LLM local peut être lent au premier lancement, car le modèle Hugging Face doit être chargé.
- La CLI et Streamlit conservent un historique court en mémoire de session pour résoudre les formulations elliptiques comme "quand sera-t-elle livrée ?" ou "et son paiement ?".
- Cet historique aide uniquement le routage sémantique; les accès SQL restent systématiquement filtrés par le `user_id` de l'utilisateur authentifié.
- L'historique n'est pas persistant : il est perdu au redémarrage de la CLI ou de la session Streamlit.
- Le bot ne modifie pas réellement les commandes; il explique la marche à suivre selon le statut.
- Les règles métier d'annulation sont simplifiées pour un contexte de formation.

## Tests Et Vérifications

Les tests pytest chargent le vrai LLM configuré dans l'environnement. Ils sont donc plus lents que des tests unitaires classiques, mais ils valident réellement le routage sémantique, la génération de réponse et les protections de l'étape 3.

Le plus simple est d'utiliser `run.sh`, qui configure le backend, vérifie les dépendances puis lance pytest :

```bash
./run.sh --mac --test
```

Sur la plateforme cible Nvidia/CUDA :

```bash
./run.sh --cuda --test
```

Sans `run.sh`, configurer d'abord le backend puis lancer pytest.

Sur Mac Apple Silicon :

```bash
export LLM_BACKEND="mlx"
export HF_LLM_MODEL_ID="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
pytest
```

Sur la plateforme cible Nvidia/CUDA :

```bash
export LLM_BACKEND="transformers"
export HF_LLM_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
export HF_DEVICE_MAP="auto"
export HF_TORCH_DTYPE="auto"
export HF_LOAD_IN_4BIT="true"
pytest
```

Avant de charger le modèle, pytest vérifie la configuration. Si `LLM_BACKEND` n'est pas défini, ou si le backend demandé ne correspond pas à la machine, pytest s'arrête immédiatement avec un message explicite. Aucun fallback CPU n'est utilisé.

Les tests vérifient notamment :

- l'accès à une commande appartenant au client après analyse par le vrai LLM;
- la réponse naturelle sur le statut d'une commande, sans statut technique SQL;
- le routage correct d'une demande de paiement;
- une réponse professionnelle même si le client est agressif;
- le blocage d'une commande appartenant à un autre client;
- le refus des demandes hors périmètre et des injections demandant les données d'autres utilisateurs.

## Correspondance Avec Les Étapes

- Etape 1 : `database.py` et `assistant.py` récupèrent les informations SQL autorisées.
- Etape 2 : `llm.py` génère une réponse professionnelle en langage naturel.
- Etape 3 : `router.py`, `assistant.py` et `database.py` assurent le routage sémantique, la protection des données et la résistance aux injections.

Les fonctions spécifiques contiennent des commentaires courts `# Etape x`, conformément à la consigne.
