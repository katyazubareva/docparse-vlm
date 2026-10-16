import json

import pytest

from docparse.data import cord_to_receipt
from docparse.metrics import aggregate, normalize, score_document
from docparse.prompts import parse_output, target_json
from docparse.schema import LineItem, Receipt

CORD_SAMPLE = {
    "menu": [
        {"nm": "Nasi Goreng", "cnt": "2", "price": "50.000"},
        {"nm": "Es Teh", "cnt": "1", "price": "8.000"},
    ],
    "sub_total": {"subtotal_price": "58.000", "tax_price": "5.800"},
    "total": {"total_price": "63.800", "cashprice": "100.000", "changeprice": "36.200"},
}


@pytest.fixture
def gold():
    return cord_to_receipt(CORD_SAMPLE)


def test_cord_normalization(gold):
    assert [i.name for i in gold.items] == ["Nasi Goreng", "Es Teh"]
    assert gold.total == "63.800" and gold.change == "36.200" and gold.tax == "5.800"
    assert gold.service is None


def test_cord_single_menu_dict_and_list_values():
    r = cord_to_receipt({"menu": {"nm": ["Ayam", "Bakar"], "price": "20.000"}, "total": {"total_price": "20.000"}})
    assert r.items == [LineItem(name="Ayam Bakar", price="20.000")]


@pytest.mark.parametrize(
    "field,a,b",
    [("total", "Rp 63.800", "63,800"), ("price", "008.000", "8000"), ("name", "  NASI  goreng", "nasi goreng"),
     ("quantity", "2x", "2")],
)
def test_normalize_equivalences(field, a, b):
    assert normalize(field, a) == normalize(field, b)


def test_perfect_prediction(gold):
    c = score_document(gold, gold)
    assert c.exact and c.tp == c.n_gold == 11 and c.total_correct


def test_item_order_does_not_matter(gold):
    shuffled = gold.model_copy(update={"items": list(reversed(gold.items))})
    assert score_document(shuffled, gold).exact


def test_invalid_prediction_scores_zero(gold):
    c = score_document(None, gold)
    assert c.tp == 0 and c.n_pred == 0 and c.n_gold == 11 and c.total_correct is False


def test_partial_and_hallucinated(gold):
    pred = Receipt(items=[LineItem(name="nasi goreng", quantity="2", price="50000"),
                          LineItem(name="Kopi", quantity="1", price="9.000")], total="63.800")
    c = score_document(pred, gold)
    assert c.tp == 3 + 1 + 1  # first item fully right, quantity "1" aligns with Es Teh, total right
    assert c.n_pred == 7


def test_aggregate_bounds(gold):
    rep = aggregate([score_document(gold, gold), score_document(None, gold)], [True, False], n_boot=200)
    assert rep.precision == 1.0 and rep.recall == 0.5
    assert rep.json_valid == 0.5 and rep.doc_exact == 0.5 and rep.total_acc == 0.5
    assert 0.0 <= rep.f1_ci95[0] <= rep.f1 <= rep.f1_ci95[1] <= 1.0


def test_parse_output_roundtrip_and_garbage(gold):
    assert parse_output("Sure! ```json\n" + target_json(gold) + "\n```") == gold
    assert parse_output("no json here") is None
    assert parse_output(json.dumps({"items": [], "bogus": 1})) is None
