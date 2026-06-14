from __future__ import annotations

import copy
import json
from typing import Any

from .schemas import Action


class MockAirlineTools:
    """Small tau-bench-like executor for tests and demos."""

    def __call__(self, data: dict[str, Any], action: Action) -> str:
        handler = getattr(self, action.name, None)
        if handler is None:
            return f"Error: unknown tool {action.name}"
        return handler(data, **action.kwargs)

    def get_user_details(self, data: dict[str, Any], user_id: str) -> str:
        user = data["users"].get(user_id)
        return json.dumps(user) if user else "Error: user not found"

    def get_reservation_details(self, data: dict[str, Any], reservation_id: str) -> str:
        reservation = data["reservations"].get(reservation_id)
        return json.dumps(reservation) if reservation else "Error: reservation not found"

    def search_direct_flight(self, data: dict[str, Any], origin: str, destination: str, date: str) -> str:
        results = []
        for flight in data["flights"].values():
            if flight["origin"] == origin and flight["destination"] == destination and date in flight["dates"]:
                row = {k: v for k, v in flight.items() if k != "dates"}
                row.update(flight["dates"][date])
                row["date"] = date
                results.append(row)
        return json.dumps(results)

    def update_reservation_flights(
        self,
        data: dict[str, Any],
        reservation_id: str,
        cabin: str,
        flights: list[dict[str, Any]],
        payment_id: str,
    ) -> str:
        reservation = data["reservations"].get(reservation_id)
        if not reservation:
            return "Error: reservation not found"
        user = data["users"][reservation["user_id"]]
        if payment_id not in user["payment_methods"]:
            return "Error: payment method not found"
        enriched = []
        for flight in flights:
            flight_info = data["flights"].get(flight["flight_number"])
            if not flight_info or flight["date"] not in flight_info["dates"]:
                return "Error: flight not found"
            enriched.append({
                "flight_number": flight["flight_number"],
                "date": flight["date"],
                "origin": flight_info["origin"],
                "destination": flight_info["destination"],
                "price": flight_info["dates"][flight["date"]]["prices"][cabin],
            })
        reservation["cabin"] = cabin
        reservation["flights"] = enriched
        reservation["payment_history"].append({"payment_id": payment_id, "amount": 20})
        return json.dumps(reservation)

    def cancel_reservation(self, data: dict[str, Any], reservation_id: str) -> str:
        reservation = data["reservations"].get(reservation_id)
        if not reservation:
            return "Error: reservation not found"
        reservation["status"] = "cancelled"
        return json.dumps(reservation)

    def update_reservation_baggages(
        self,
        data: dict[str, Any],
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> str:
        reservation = data["reservations"].get(reservation_id)
        if not reservation:
            return "Error: reservation not found"
        reservation["total_baggages"] = total_baggages
        reservation["nonfree_baggages"] = nonfree_baggages
        reservation["payment_history"].append({"payment_id": payment_id, "amount": 50})
        return json.dumps(reservation)

    def book_reservation(self, data: dict[str, Any], **kwargs: Any) -> str:
        reservation_id = "HATHAT"
        reservation = {"reservation_id": reservation_id, **copy.deepcopy(kwargs)}
        data["reservations"][reservation_id] = reservation
        data["users"][kwargs["user_id"]]["reservations"].append(reservation_id)
        return json.dumps(reservation)

    def send_certificate(self, data: dict[str, Any], user_id: str, amount: int) -> str:
        user = data["users"].get(user_id)
        if not user:
            return "Error: user not found"
        payment_id = "certificate_3221322"
        user["payment_methods"][payment_id] = {"source": "certificate", "amount": amount, "id": payment_id}
        return f"Certificate {payment_id} added to user {user_id} with amount {amount}."


def make_demo_data() -> dict[str, Any]:
    return {
        "users": {
            "olivia_gonzalez_2305": {
                "user_id": "olivia_gonzalez_2305",
                "reservations": ["Z7GOZK"],
                "payment_methods": {
                    "credit_card_1111111": {"source": "credit_card", "id": "credit_card_1111111"},
                },
            }
        },
        "reservations": {
            "Z7GOZK": {
                "reservation_id": "Z7GOZK",
                "user_id": "olivia_gonzalez_2305",
                "status": "confirmed",
                "cabin": "business",
                "flights": [
                    {
                        "flight_number": "HAT100",
                        "date": "2024-05-20",
                        "origin": "AUS",
                        "destination": "EWR",
                        "price": 500,
                    }
                ],
                "payment_history": [{"payment_id": "credit_card_1111111", "amount": 500}],
                "total_baggages": 0,
                "nonfree_baggages": 0,
            }
        },
        "flights": {
            "HAT200": {
                "flight_number": "HAT200",
                "origin": "AUS",
                "destination": "EWR",
                "dates": {
                    "2024-05-21": {
                        "status": "available",
                        "prices": {"economy": 180, "business": 400},
                    }
                },
            }
        },
    }

