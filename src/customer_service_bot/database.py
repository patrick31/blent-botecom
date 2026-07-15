"""Accès contrôlé à la base SQLite des commandes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from customer_service_bot.config import AuthenticatedUser


@dataclass(frozen=True)
class Order:
    """Commande rattachée à un utilisateur."""

    order_id: int
    user_id: int
    status: str
    date_purchase: str | None
    date_shipped: str | None
    date_delivered: str | None

    @property
    def status_label(self) -> str:
        """Statut métier formulé en français."""

        labels = {
            "invoiced": "validée et payée, en attente d'expédition",
            "shipped": "expédiée, en cours d'acheminement",
            "delivered": "livrée",
        }
        return labels.get(self.status, self.status)


class OrderRepository:
    """Repository SQLite avec filtrage systématique par utilisateur authentifié."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    # Etape 1
    def get_user_by_email(self, email: str) -> AuthenticatedUser | None:
        """Récupère un utilisateur par email pour simuler l'authentification."""

        query = """
        SELECT user_id, first_name, last_name, email
        FROM users
        WHERE lower(email) = lower(?)
        LIMIT 1
        """
        with self._connect() as connection:
            row = connection.execute(query, (email,)).fetchone()
        if row is None:
            return None
        return AuthenticatedUser(
            user_id=int(row["user_id"]),
            first_name=str(row["first_name"]),
            last_name=str(row["last_name"]),
            email=str(row["email"]),
        )

    # Etape 1
    def list_users(self) -> list[AuthenticatedUser]:
        """Liste les utilisateurs disponibles pour les tests CLI."""

        query = """
        SELECT user_id, first_name, last_name, email
        FROM users
        ORDER BY user_id
        """
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [
            AuthenticatedUser(
                user_id=int(row["user_id"]),
                first_name=str(row["first_name"]),
                last_name=str(row["last_name"]),
                email=str(row["email"]),
            )
            for row in rows
        ]

    # Etape 1 (modifié étape 3)
    def list_orders_for_user(self, user: AuthenticatedUser) -> list[Order]:
        """Liste uniquement les commandes du client authentifié."""

        # ajout Etape 3: le filtre user_id est imposé par le code, pas par le prompt.
        query = """
        SELECT order_id, user_id, status, date_purchase, date_shipped, date_delivered
        FROM orders
        WHERE user_id = ?
        ORDER BY date_purchase DESC
        """
        with self._connect() as connection:
            rows = connection.execute(query, (user.user_id,)).fetchall()
        return [self._row_to_order(row) for row in rows]

    # Etape 1 (modifié étape 3)
    def get_order_for_user(self, user: AuthenticatedUser, order_id: int) -> Order | None:
        """Récupère une commande si elle appartient au client authentifié."""

        # ajout Etape 3: empêche un client de consulter une commande d'un autre client.
        query = """
        SELECT order_id, user_id, status, date_purchase, date_shipped, date_delivered
        FROM orders
        WHERE user_id = ? AND order_id = ?
        LIMIT 1
        """
        with self._connect() as connection:
            row = connection.execute(query, (user.user_id, order_id)).fetchone()
        return self._row_to_order(row) if row else None

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        return Order(
            order_id=int(row["order_id"]),
            user_id=int(row["user_id"]),
            status=str(row["status"]),
            date_purchase=row["date_purchase"],
            date_shipped=row["date_shipped"],
            date_delivered=row["date_delivered"],
        )


def format_orders_for_prompt(orders: Iterable[Order]) -> str:
    """Transforme une liste de commandes en contexte compact pour le LLM."""

    lines = []
    for order in orders:
        if order.status == "delivered":
            state = f"elle est deja livree depuis le {order.date_delivered}"
        elif order.status == "shipped":
            state = f"elle est expediee depuis le {order.date_shipped} et n'est pas encore livree"
        elif order.status == "invoiced":
            state = "elle est validee et payee, mais pas encore expediee"
        else:
            state = f"son statut exact est {order.status_label}"

        lines.append(
            "- Commande #{order_id}: {state}. Date d'achat: {purchase}. "
            "Date d'expedition: {shipped}. Date de livraison effective: {delivered}.".format(
                order_id=order.order_id,
                state=state,
                purchase=order.date_purchase or "non renseignee",
                shipped=order.date_shipped or "non renseignee",
                delivered=order.date_delivered or "non renseignee",
            )
        )
    return "\n".join(lines) if lines else "Aucune commande trouvee."
