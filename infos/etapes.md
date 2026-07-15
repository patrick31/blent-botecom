Étapes du projet

Etape 1 : Récupération des informations

La première étape consiste à récupérer des informations disponibles dans la base de données SQL lorsque l'utilisateur pose une question au sujet d'une commande passée ou en cours. Pour cela, le bot doit être en mesurer de pouvoir **requêter sur la base SQL** afin d'obtenir l'informations en temps réel.

La réponse doit être formatée en **langage naturel** : en particulier, si la question concerne le statut d'une commande, il est nécessaire d'indiquer le statut non pas dont la manière où elle est encodée (exemple `invoiced`), mais de manière compréhensible.

Etape 2 : Comportement du bot

La deuxième étape consiste à mettre en place le comportement du bot, notamment pour qu'il puisse répondre à la questions de l'utilisateur en fonction de la nature de celle-ci. On veillera à apporter tous les ajustements nécessaires pour répondre avec les mots justes à chaque demande de l'utilisateur, même dans le cas où celui-ci deviendrait agressif.

Etape 3 : Protection des données et routage sémantique

La dernière étape cherche à mettre en place une protection des données sur le bot, notamment pour éviter les **injections de prompts** qui permettraient à l'utilisateur authentifié d'accéder à des informations sur des commandes d'autres utilisateurs.

On limitera également les réponses du bot au sujet du service client, et toutes les demandes qui ne concernent pas ce sujet seront rejetées.



