"""Interface en ligne de commande pour tester l'assistant."""

from __future__ import annotations

import argparse

from customer_service_bot.assistant import CustomerServiceAssistant
from customer_service_bot.config import AppConfig
from customer_service_bot.database import OrderRepository


def build_parser() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments de la CLI."""

    parser = argparse.ArgumentParser(description="Assistant de service client pour le suivi de commandes.")
    parser.add_argument("--email", help="Email de l'utilisateur authentifie. Si absent, affiche un menu de selection.")
    parser.add_argument("--question", help="Question a poser au bot. Si absent, demarre un mode interactif.")
    parser.add_argument("--use-llm", action="store_true", help="Option conservee pour compatibilite: le LLM est toujours utilise.")
    return parser


def select_user_for_cli(repository: OrderRepository, email: str | None):
    """Retourne l'utilisateur choisi par email ou via un menu interactif."""

    if email:
        user = repository.get_user_by_email(email)
        if user is None:
            raise SystemExit("Utilisateur introuvable dans la base.")
        return user

    users = repository.list_users()
    if not users:
        raise SystemExit("Aucun utilisateur disponible dans la base.")

    print("Choisissez un utilisateur pour tester l'assistant :")
    for index, user in enumerate(users, start=1):
        print(f"{index}. {user.first_name} {user.last_name} <{user.email}>")

    while True:
        choice = input("Numero utilisateur > ").strip()
        if choice.lower() in {"exit", "quit"}:
            raise SystemExit(0)
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(users):
                return users[selected_index - 1]
        print(f"Choix invalide. Entrez un numero entre 1 et {len(users)}, ou 'exit' pour quitter.")


def main() -> None:
    """Point d'entrée de la commande."""

    args = build_parser().parse_args()
    config = AppConfig.from_env()
    config = AppConfig(
        database_path=config.database_path,
        llm_model_id=config.llm_model_id,
        embedding_model_id=config.embedding_model_id,
        max_new_tokens=config.max_new_tokens,
        analysis_max_new_tokens=config.analysis_max_new_tokens,
        temperature=config.temperature,
        llm_backend=config.llm_backend,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        load_in_4bit=config.load_in_4bit,
        use_llm=True,
    )

    repository = OrderRepository(config.database_path)
    user = select_user_for_cli(repository, args.email)

    assistant = CustomerServiceAssistant(config=config, repository=repository)
    history: list[tuple[str, str]] = []
    if args.question:
        answer = assistant.answer(user, args.question, history=history)
        print(answer)
        return

    print("Assistant pret. Tapez 'exit' pour quitter.")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer = assistant.answer(user, question, history=history)
            print(answer)
            history.append((question, answer))


if __name__ == "__main__":
    main()
