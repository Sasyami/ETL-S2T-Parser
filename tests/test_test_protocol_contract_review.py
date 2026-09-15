import json

from agents.test_protocol import RawTestProtocolContract, RawTestProtocolLoad
from agents.test_protocol_contract_review import (
    VALIDATION_CONTRACT_REVIEW_ISSUE_CODES,
    VALIDATION_CONTRACT_REVIEW_PROMPT,
    VALIDATION_CONTRACT_REVIEW_TOOL_NAME,
    render_validation_contract_review_request,
    validate_validation_contract_review,
    validation_contract_review_tool_schema,
)


def _candidate() -> RawTestProtocolContract:
    return RawTestProtocolContract(
        file_scope_kind="file_mention",
        file_mention="Согласование витрины.xlsx",
        loads=[
            RawTestProtocolLoad(
                source_mentions=["stage.customer_delta"],
                target_mention="dm.customer",
                requested_checks=["row_count", "key_uniqueness"],
                explicit_key=["customer_id"],
            )
        ],
        requested_checks=[],
        mode="explicit",
    )


def test_accept_review_is_valid() -> None:
    result = validate_validation_contract_review(
        {
            "decision": "accept",
            "issues": [],
        }
    )

    assert result.status == "valid"
    assert result.review is not None
    assert result.review.decision == "accept"
    assert result.review.issues == []
    assert result.errors == []
    assert result.silent_fallback is False


def test_repair_accepts_every_bounded_issue_code() -> None:
    payload = {
        "decision": "repair",
        "issues": [
            {
                "code": code,
                "location": f"loads[{index}]",
                "message": f"Исправить расхождение {index}.",
            }
            for index, code in enumerate(
                VALIDATION_CONTRACT_REVIEW_ISSUE_CODES
            )
        ],
    }

    result = validate_validation_contract_review(payload)

    assert result.status == "valid"
    assert result.review is not None
    assert result.review.decision == "repair"
    assert [issue.code for issue in result.review.issues] == list(
        VALIDATION_CONTRACT_REVIEW_ISSUE_CODES
    )


def test_malformed_review_is_invalid() -> None:
    result = validate_validation_contract_review(
        {
            "decision": "accept",
            "issues": [],
            "unexpected": True,
        }
    )

    assert result.status == "invalid"
    assert result.review is None
    assert result.errors
    assert result.silent_fallback is False


def test_unknown_repair_issue_is_invalid() -> None:
    result = validate_validation_contract_review(
        {
            "decision": "repair",
            "issues": [
                {
                    "code": "model_invented_issue",
                    "location": "loads[0]",
                    "message": "Свободный код запрещён.",
                }
            ],
        }
    )

    assert result.status == "invalid"
    assert result.review is None
    assert any(error.location.endswith("code") for error in result.errors)


def test_empty_repair_is_invalid() -> None:
    result = validate_validation_contract_review(
        {
            "decision": "repair",
            "issues": [],
        }
    )

    assert result.status == "invalid"
    assert result.review is None
    assert result.errors


def test_accept_with_repair_issues_is_invalid() -> None:
    result = validate_validation_contract_review(
        {
            "decision": "accept",
            "issues": [
                {
                    "code": "missing_file_scope",
                    "location": "file_scope_kind",
                    "message": "Файл потерян.",
                }
            ],
        }
    )

    assert result.status == "invalid"
    assert result.review is None


def test_renderer_separates_authoritative_task_from_candidate_data() -> None:
    original_task = (
        "Сформируй explicit-протокол для файла Согласование витрины.xlsx: "
        "stage.customer_delta -> dm.customer, row_count и key_uniqueness."
    )

    rendered = render_validation_contract_review_request(
        original_task,
        _candidate(),
    )

    marker, encoded = rendered.split("\n", 1)
    payload = json.loads(encoded)
    assert marker == "REVIEW_INPUT:"
    assert payload["authoritative_user_request"] == {
        "original_task": original_task
    }
    assert payload["candidate_data"] == _candidate().model_dump(mode="json")
    assert "authoritative_user_request.original_task" in (
        VALIDATION_CONTRACT_REVIEW_PROMPT
    )
    assert "candidate_data" in VALIDATION_CONTRACT_REVIEW_PROMPT


def test_tool_schema_is_flat_closed_and_gigachat_safe() -> None:
    from langchain_gigachat.utils.function_calling import gigachat_fix_schema

    schema = validation_contract_review_tool_schema()
    serialized = json.dumps(schema, ensure_ascii=False)
    function = schema["function"]
    parameters = function["parameters"]

    assert function["name"] == VALIDATION_CONTRACT_REVIEW_TOOL_NAME
    assert parameters["required"] == ["decision", "issues"]
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["issues"]["items"][
        "additionalProperties"
    ] is False
    assert parameters["properties"]["issues"]["items"]["properties"][
        "code"
    ]["enum"] == list(VALIDATION_CONTRACT_REVIEW_ISSUE_CODES)
    for unsupported_keyword in ("$ref", "$defs", "anyOf", "oneOf", "allOf"):
        assert unsupported_keyword not in serialized
    assert gigachat_fix_schema(schema) == schema
