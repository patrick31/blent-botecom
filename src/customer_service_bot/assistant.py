"""Orchestration de l'assistant de service client."""

from __future__ import annotations

from typing import Any

from customer_service_bot.config import AppConfig, AuthenticatedUser
from customer_service_bot.database import Order, OrderRepository, format_orders_for_prompt
from customer_service_bot.llm import (
    HuggingFaceLLMFactory,
    LangChainHuggingFaceResponseGenerator,
    ResponseContext,
)
from customer_service_bot.router import LangChainQuestionAnalyzer, QueryAnalysis


class CustomerServiceAssistant:
    """Assistant conversationnel LLM limité au suivi des commandes."""

    def __init__(
        self,
        config: AppConfig,
        repository: OrderRepository | None = None,
        question_analyzer: Any | None = None,
        response_generator: Any | None = None,
    ) -> None:
        self.config = config
        self.repository = repository or OrderRepository(config.database_path)
        llm_factory = HuggingFaceLLMFactory(config)
        self.question_analyzer = question_analyzer or LangChainQuestionAnalyzer(llm_factory)
        self.response_generator = response_generator or LangChainHuggingFaceResponseGenerator(llm_factory)

    # Etape 2 (modifié étape 3)
    def answer(self, user: AuthenticatedUser, message: str, history: list[tuple[str, str]] | None = None) -> str:
        """Répond à une question utilisateur libre avec analyse et réponse LLM."""

        history_text = format_conversation_history(history or [])
        analysis = self.question_analyzer.analyze(user, message, history=history_text)

        # ajout Etape 3: le routage LLM bloque les demandes hors service client avant tout accès SQL.
        if not analysis.is_customer_service:
            facts = build_refusal_facts(analysis)
            return self.response_generator.generate(
                ResponseContext(user=user, question=message, facts=facts, analysis=analysis, orders=[], history=history_text)
            )

        if analysis.intent == "presentation":
            facts = build_presentation_facts()
            return self.response_generator.generate(
                ResponseContext(user=user, question=message, facts=facts, analysis=analysis, orders=[], history=history_text)
            )

        orders = self._load_authorized_orders(user, analysis)
        facts = build_facts(analysis, orders)
        return self.response_generator.generate(
            ResponseContext(user=user, question=message, facts=facts, analysis=analysis, orders=orders, history=history_text)
        )

    # Etape 1 (initialement basique sans controle authorisation et modifié étape 3 pour ajout des contrôles)
    def _load_authorized_orders(self, user: AuthenticatedUser, analysis: QueryAnalysis) -> list[Order]:
        """Charge uniquement les commandes autorisées par le compte authentifié."""

        # ajout Etape 3: la derniere commande est calculee uniquement dans le perimetre du compte courant.
        if analysis.order_id is None and analysis.asks_for_latest_order:
            orders = self.repository.list_orders_for_user(user)
            return orders[:1]

        # ajout Etape 3: on liste les commandes seulement si le client le demande explicitement.
        if analysis.order_id is None and not analysis.asks_for_all_orders:
            return []

        # ajout Etape 3: en absence d'order_id, on liste seulement les commandes du compte courant.
        if analysis.order_id is None:
            return self.repository.list_orders_for_user(user)

        # ajout Etape 3: même si le LLM extrait un order_id, la requête SQL impose user_id.
        order = self.repository.get_order_for_user(user, analysis.order_id)
        return [order] if order else []


def format_conversation_history(history: list[tuple[str, str]], max_turns: int = 4) -> str:
    """Formate les derniers échanges utiles pour le routage LLM."""

    if not history:
        return "Aucun."

    turns = history[-max_turns:]
    lines: list[str] = []
    for index, (question, answer) in enumerate(turns, start=1):
        lines.append(f"Tour {index} - client: {question}")
        lines.append(f"Tour {index} - assistant: {answer}")
    return "\n".join(lines)


# Etape 1 (modifié étape 2)
def build_facts(analysis: QueryAnalysis, orders: list[Order]) -> str:
    """Prépare les faits SQL autorisés que le LLM peut reformuler."""

    if not orders:
        # ajout Etape 3: ne pas révéler si une commande existe pour un autre utilisateur.
        if analysis.order_id is not None:
            return (
                f"Aucune commande #{analysis.order_id} n'est rattachee au compte authentifie. "
                "Par securite, l'assistant ne peut consulter que les commandes de cet utilisateur."
            )
        return "Aucune commande rattachee au compte authentifie n'a ete trouvee."

    facts = [format_orders_for_prompt(orders)]
    if analysis.intent == "modification_annulation":
        facts.append(build_change_policy(orders))
    if analysis.intent == "paiement":
        facts.append("Un statut invoiced, shipped ou delivered indique une commande deja validee et payee.")
    return "\n".join(facts)


# Etape 2 (modifié étape 3)
def build_presentation_facts() -> str:
    """Prépare le message de présentation lorsque la demande n'est pas précise."""

    return (
        "Le client n'a pas encore formule de demande precise sur une commande. "
        "L'assistant doit se presenter brievement et expliquer qu'il peut aider sur le statut, "
        "la livraison, le paiement, la modification ou l'annulation de commandes deja passees. "
        "L'assistant doit inviter le client a indiquer un numero de commande ou a poser sa question."
    )


# Etape 3
def build_refusal_facts(analysis: QueryAnalysis) -> str:
    """Prépare un refus que le LLM doit reformuler poliment."""

    reason = analysis.refusal_reason or "La demande est hors du perimetre du service client commandes."
    return (
        f"Demande refusee par le routeur LLM: {reason}. "
        "L'assistant peut uniquement aider sur les commandes deja passees du client authentifie."
    )


# Etape 2
def build_change_policy(orders: list[Order]) -> str:
    """Explique les règles de modification ou d'annulation selon le statut."""

    messages = []
    for order in orders:
        if order.status == "invoiced":
            policy = "la commande n'est pas encore expediee; une demande de modification ou d'annulation peut etre transmise"
        elif order.status == "shipped":
            policy = "la commande est deja expediee; l'annulation directe n'est plus disponible"
        elif order.status == "delivered":
            policy = "la commande est livree; il faut passer par une demande de retour ou le service client"
        else:
            policy = "le statut doit etre verifie par un conseiller"
        messages.append(f"Commande #{order.order_id}: {policy}.")
    return "\n".join(messages)
