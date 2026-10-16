"""Target schema shared by every system: gold labels, baselines and the fine-tuned model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

HEADER_FIELDS = ("subtotal", "tax", "service", "discount", "total", "cash", "change")
ITEM_FIELDS = ("name", "quantity", "price")
AMOUNT_FIELDS = frozenset({"price", *HEADER_FIELDS})


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    quantity: str | None = None
    price: str | None = Field(None, description="Line total as printed on the receipt.")


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[LineItem] = Field(default_factory=list)
    subtotal: str | None = None
    tax: str | None = None
    service: str | None = None
    discount: str | None = None
    total: str | None = None
    cash: str | None = None
    change: str | None = None
