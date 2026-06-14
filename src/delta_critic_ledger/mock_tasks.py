from __future__ import annotations

import copy
from dataclasses import dataclass

from .mock_airline import make_demo_data
from .schemas import Action


@dataclass(frozen=True)
class MockTask:
    task_id: str
    initial_data: dict
    target_actions: list[Action]
    trajectories: dict[str, list[Action]]
    seed_entities: dict[str, list[str]]


def cancel_reservation_task() -> MockTask:
    data = make_demo_data()
    return MockTask(
        task_id="cancel_reservation",
        initial_data=data,
        target_actions=[Action("cancel_reservation", {"reservation_id": "Z7GOZK"})],
        seed_entities={"user_id": ["olivia_gonzalez_2305"]},
        trajectories={
            "successful": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action("cancel_reservation", {"reservation_id": "Z7GOZK"}),
            ],
            "wrong_write": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("cancel_reservation", {"reservation_id": "BAD999"}),
            ],
            "premature_write": [
                Action("cancel_reservation", {"reservation_id": "Z7GOZK"}),
            ],
            "noop_then_success": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action("cancel_reservation", {"reservation_id": "Z7GOZK"}),
            ],
        },
    )


def baggage_update_task() -> MockTask:
    data = make_demo_data()
    return MockTask(
        task_id="baggage_update",
        initial_data=data,
        target_actions=[
            Action(
                "update_reservation_baggages",
                {
                    "reservation_id": "Z7GOZK",
                    "total_baggages": 2,
                    "nonfree_baggages": 1,
                    "payment_id": "credit_card_1111111",
                },
            )
        ],
        seed_entities={"user_id": ["olivia_gonzalez_2305"]},
        trajectories={
            "successful": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action(
                    "update_reservation_baggages",
                    {
                        "reservation_id": "Z7GOZK",
                        "total_baggages": 2,
                        "nonfree_baggages": 1,
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
            "wrong_write": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action(
                    "update_reservation_baggages",
                    {
                        "reservation_id": "BAD999",
                        "total_baggages": 2,
                        "nonfree_baggages": 1,
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
            "premature_write": [
                Action(
                    "update_reservation_baggages",
                    {
                        "reservation_id": "Z7GOZK",
                        "total_baggages": 2,
                        "nonfree_baggages": 1,
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
            "noop_then_success": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action(
                    "update_reservation_baggages",
                    {
                        "reservation_id": "Z7GOZK",
                        "total_baggages": 2,
                        "nonfree_baggages": 1,
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
        },
    )


def flight_update_task() -> MockTask:
    data = make_demo_data()
    return MockTask(
        task_id="flight_update",
        initial_data=data,
        target_actions=[
            Action(
                "update_reservation_flights",
                {
                    "reservation_id": "Z7GOZK",
                    "cabin": "economy",
                    "flights": [{"flight_number": "HAT200", "date": "2024-05-21"}],
                    "payment_id": "credit_card_1111111",
                },
            )
        ],
        seed_entities={"user_id": ["olivia_gonzalez_2305"]},
        trajectories={
            "successful": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action("search_direct_flight", {"origin": "AUS", "destination": "EWR", "date": "2024-05-21"}),
                Action(
                    "update_reservation_flights",
                    {
                        "reservation_id": "Z7GOZK",
                        "cabin": "economy",
                        "flights": [{"flight_number": "HAT200", "date": "2024-05-21"}],
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
            "wrong_write": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action(
                    "update_reservation_flights",
                    {
                        "reservation_id": "Z7GOZK",
                        "cabin": "economy",
                        "flights": [{"flight_number": "BAD123", "date": "2024-05-21"}],
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
            "premature_write": [
                Action(
                    "update_reservation_flights",
                    {
                        "reservation_id": "Z7GOZK",
                        "cabin": "economy",
                        "flights": [{"flight_number": "HAT200", "date": "2024-05-21"}],
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
            "noop_then_success": [
                Action("get_user_details", {"user_id": "olivia_gonzalez_2305"}),
                Action("get_reservation_details", {"reservation_id": "Z7GOZK"}),
                Action("search_direct_flight", {"origin": "AUS", "destination": "EWR", "date": "2024-05-21"}),
                Action("search_direct_flight", {"origin": "AUS", "destination": "EWR", "date": "2024-05-21"}),
                Action(
                    "update_reservation_flights",
                    {
                        "reservation_id": "Z7GOZK",
                        "cabin": "economy",
                        "flights": [{"flight_number": "HAT200", "date": "2024-05-21"}],
                        "payment_id": "credit_card_1111111",
                    },
                ),
            ],
        },
    )


def get_task_registry() -> dict[str, MockTask]:
    tasks = [cancel_reservation_task(), baggage_update_task(), flight_update_task()]
    return {task.task_id: task for task in tasks}

