"""Model-owned extraction contract for the closed SQL-risk scope.

The module intentionally does not call an LLM and does not discover literals
in user text.  A caller asks its LLM for :class:`SqlRiskScopeExtraction`, then
passes that native-call payload here.  Validation only proves that the model's
verbatim attestations occur at exact, structurally valid locations in the
original task.

There is no entity resolution, natural-language classification, regex search,
candidate enumeration or fallback to the general agentic pipeline here.  An
invalid result is returned without a contract.  The caller may use the clear
issues for at most one repair call.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)


SQL_RISK_SCOPE_EXTRACTION_TOOL_NAME = "submit_sql_risk_scope"
MAX_SQL_RISK_SCOPE_EXTRACTION_ATTEMPTS = 2
SQL_RISK_SCOPE_EXECUTION_MODES = (
    "row_filtering",
    "conditional_cardinality",
    "nullable_constraint",
    "value_changes",
    "write_semantics",
)

SqlRiskScopeExecutionMode = Literal[
    "row_filtering",
    "conditional_cardinality",
    "nullable_constraint",
    "value_changes",
    "write_semantics",
]
SqlRiskScopeExtractionStatus = Literal["valid", "invalid"]
SqlRiskScopeExtractionIssueCode = Literal[
    "schema_error",
    "endpoint_mismatch",
    "endpoint_not_found",
    "endpoint_not_bounded",
    "file_id_mismatch",
    "file_id_not_found",
    "file_id_not_bounded",
]

_TABLE_MODES = frozenset(
    {
        "row_filtering",
        "conditional_cardinality",
        "write_semantics",
    }
)
_FIELD_MODES = frozenset({"nullable_constraint", "value_changes"})


class SqlRiskScopeEndpoint(BaseModel):
    """One model-selected endpoint with an explicit table/field role."""

    model_config = ConfigDict(extra="forbid")

    table_name: StrictStr = Field(
        min_length=1,
        description=(
            "Technical table name copied exactly from original_task."
        ),
    )
    field_name: StrictStr | None = Field(
        default=None,
        description=(
            "Technical field name copied exactly from original_task; omit "
            "this field for a table-level mode."
        ),
    )

    @field_validator("table_name", "field_name")
    @classmethod
    def _require_exact_identifier_component(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("technical names must not contain outer/inner whitespace")
        if "→" in value or "`" in value:
            raise ValueError("technical names must not contain arrows or quote markers")
        return value

    @property
    def literal(self) -> str:
        """Return the exact unquoted endpoint represented by this schema."""

        if self.field_name is None:
            return self.table_name
        return f"{self.table_name}.{self.field_name}"


class SqlRiskScopeEndpointAttestation(BaseModel):
    """Verbatim endpoint components selected by the extraction LLM."""

    model_config = ConfigDict(extra="forbid")

    table_name: StrictStr = Field(
        min_length=1,
        description=(
            "Exact table-name component copied from original_task."
        ),
    )
    field_name: StrictStr | None = Field(
        default=None,
        description=(
            "Exact field-name component copied from original_task; omit for "
            "a table-level mode."
        ),
    )


class SqlRiskScopeOriginAttestation(BaseModel):
    """Verbatim component attestations selected from ``original_task``."""

    model_config = ConfigDict(extra="forbid")

    source: SqlRiskScopeEndpointAttestation
    target: SqlRiskScopeEndpointAttestation
    file_id: StrictStr | None = Field(
        default=None,
        description=(
            "Exact decimal token for the selected file_id copied from "
            "original_task; omit this field when file_id is absent."
        ),
    )


class SqlRiskScopeExtraction(BaseModel):
    """The sole native LLM output schema for closed SQL-risk extraction."""

    model_config = ConfigDict(extra="forbid")

    execution_mode: SqlRiskScopeExecutionMode = Field(
        description=(
            "Choose exactly one closed SQL-risk mode requested by the user."
        )
    )
    source: SqlRiskScopeEndpoint
    target: SqlRiskScopeEndpoint
    file_id: Annotated[StrictInt, Field(ge=1)] | None = Field(
        default=None,
        description=(
            "Positive literal file_id from original_task; omit when the "
            "selected mode does not require and the task does not supply it."
        ),
    )
    origin: SqlRiskScopeOriginAttestation

    @model_validator(mode="after")
    def _require_mode_shape(self) -> "SqlRiskScopeExtraction":
        source_has_field = self.source.field_name is not None
        target_has_field = self.target.field_name is not None
        if source_has_field != target_has_field:
            raise ValueError(
                "source and target must both be tables or both be fields"
            )
        if self.execution_mode in _TABLE_MODES and source_has_field:
            raise ValueError(
                f"{self.execution_mode} requires table-level endpoints"
            )
        if self.execution_mode in _FIELD_MODES and not source_has_field:
            raise ValueError(
                f"{self.execution_mode} requires field-level endpoints"
            )
        if self.execution_mode == "nullable_constraint" and self.file_id is None:
            raise ValueError("nullable_constraint requires a literal file_id")
        if (
            (self.origin.source.field_name is None) != (not source_has_field)
            or (self.origin.target.field_name is None) != (not target_has_field)
        ):
            raise ValueError(
                "origin endpoint components must match the selected scope shape"
            )
        if (self.file_id is None) != (self.origin.file_id is None):
            raise ValueError(
                "file_id and origin.file_id must either both be present or both be null"
            )
        return self


class SqlRiskScopeExtractionIssue(BaseModel):
    """One repairable schema or exact-origin validation failure."""

    model_config = ConfigDict(extra="forbid")

    code: SqlRiskScopeExtractionIssueCode
    location: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SqlRiskScopeOriginLocations(BaseModel):
    """Exact source locations proven by the non-extracting validator."""

    model_config = ConfigDict(extra="forbid")

    source_table_start: int = Field(ge=0)
    source_table_end: int = Field(gt=0)
    source_field_start: int | None = Field(default=None, ge=0)
    source_field_end: int | None = Field(default=None, gt=0)
    target_table_start: int = Field(ge=0)
    target_table_end: int = Field(gt=0)
    target_field_start: int | None = Field(default=None, ge=0)
    target_field_end: int | None = Field(default=None, gt=0)
    file_id_start: int | None = Field(default=None, ge=0)
    file_id_end: int | None = Field(default=None, gt=0)


class SqlRiskScopeExtractionValidation(BaseModel):
    """Fail-closed validation boundary consumed by the scope pipeline."""

    model_config = ConfigDict(extra="forbid")

    status: SqlRiskScopeExtractionStatus
    contract: SqlRiskScopeExtraction | None = None
    origin_locations: SqlRiskScopeOriginLocations | None = None
    issues: list[SqlRiskScopeExtractionIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_consistent_status(self) -> "SqlRiskScopeExtractionValidation":
        if self.status == "valid":
            if self.contract is None or self.origin_locations is None or self.issues:
                raise ValueError("valid extraction requires contract and locations only")
        elif self.contract is not None or self.origin_locations is not None or not self.issues:
            raise ValueError("invalid extraction requires issues and exposes no contract")
        return self


SQL_RISK_SCOPE_EXTRACTION_PROMPT = f"""
Ты extractor отдельного SQL-risk scope. Верни ровно один native call
`{SQL_RISK_SCOPE_EXTRACTION_TOOL_NAME}` по схеме `SqlRiskScopeExtraction`.

Выбери один режим внутри этого pipeline:
- `row_filtering` — риск потери строк для одной пары таблиц;
- `conditional_cardinality` — риск размножения строк для одной пары таблиц;
- `nullable_constraint` — совместимость NULL/NOT NULL одной пары полей;
- `value_changes` — механизм изменения значения одной пары полей;
- `write_semantics` — сохранённая write-стратегия одной пары таблиц.

Не анализируй SQL и не отвечай пользователю. Скопируй source/target и optional
`file_id` только из `original_task`. В `origin.source` и `origin.target`
скопируй `table_name` и, для field-mode, `field_name` отдельными минимальными
точными компонентами дословно, включая регистр. Они могут встречаться как
единый `table.field` либо раздельно. Связь может быть сформулирована как угодно:
не требуй стрелку или фиксированные слова. В `origin.file_id`
скопируй только точный цифровой token выбранного file_id; если file_id не задан,
не возвращай оба optional поля. `stable_context` используй только для смысла
терминов и выбора режима; source/target/file_id и их attestations из него брать
запрещено. Не исправляй опечатки, не нормализуй имена и не выбирай приближённые
кандидаты. Если закрытый контракт невозможно заполнить буквально, native call
останется невалидным и pipeline завершится unavailable без agentic fallback.
""".strip()


def _issue(
    code: SqlRiskScopeExtractionIssueCode,
    location: str,
    message: str,
) -> SqlRiskScopeExtractionIssue:
    return SqlRiskScopeExtractionIssue(
        code=code,
        location=location,
        message=message,
    )


def _exact_locations(text: str, quote: str) -> tuple[tuple[int, int], ...]:
    """Locate only the supplied exact quote; never discover alternatives."""

    locations: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(quote, cursor)
        if start < 0:
            break
        end = start + len(quote)
        locations.append((start, end))
        cursor = start + 1
    return tuple(locations)


def _left_is_bounded(text: str, start: int) -> bool:
    if start == 0:
        return True
    previous = text[start - 1]
    if previous.isalnum() or previous in "_$./-":
        return False
    if previous == ":" and start >= 2 and text[start - 2] == ":":
        return False
    return True


def _right_is_bounded(text: str, end: int) -> bool:
    if end == len(text):
        return True
    following = text[end]
    if following.isalnum() or following in "_$/":
        return False
    if following in ".-:":
        after_punctuation = text[end + 1] if end + 1 < len(text) else ""
        if (
            (following == ":" and after_punctuation == ":")
            or after_punctuation.isalnum()
            or (
                bool(after_punctuation)
                and after_punctuation in "_$./-"
            )
        ):
            return False
    return True


def _bounded_exact_locations(
    text: str,
    quote: str,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, end)
        for start, end in _exact_locations(text, quote)
        if _left_is_bounded(text, start) and _right_is_bounded(text, end)
    )


def _component_surface_matches(surface: str, component: str) -> bool:
    return surface == component or surface == f"`{component}`"


def _bounded_component_locations(
    text: str,
    quote: str,
    *,
    left_component: str | None = None,
    right_component: str | None = None,
) -> tuple[tuple[int, int], ...]:
    """Validate a selected component, including an exact ``table.field``."""

    locations: list[tuple[int, int]] = []
    for start, end in _exact_locations(text, quote):
        left_is_valid = _left_is_bounded(text, start) or bool(
            left_component
            and text[:start].endswith(f"{left_component}.")
        )
        right_is_valid = _right_is_bounded(text, end) or bool(
            right_component
            and text[end:].startswith(f".{right_component}")
        )
        if left_is_valid and right_is_valid:
            locations.append((start, end))
    return tuple(locations)


def _validate_component_attestation(
    *,
    component: str,
    quote: str,
    location: str,
    original_task: str,
    left_component: str | None = None,
    right_component: str | None = None,
) -> tuple[tuple[int, int] | None, list[SqlRiskScopeExtractionIssue]]:
    """Verify one model-selected component without discovering alternatives."""

    if quote != quote.strip() or not _component_surface_matches(
        quote,
        component,
    ):
        return None, [
            _issue(
                "endpoint_mismatch",
                location,
                "attestation must exactly match the selected component",
            )
        ]
    locations = _exact_locations(original_task, quote)
    if not locations:
        return None, [
            _issue(
                "endpoint_not_found",
                location,
                "verbatim component attestation is absent from original_task",
            )
        ]
    bounded = _bounded_component_locations(
        original_task,
        quote,
        left_component=left_component,
        right_component=right_component,
    )
    if not bounded:
        return None, [
            _issue(
                "endpoint_not_bounded",
                location,
                "attested component is only a partial token in original_task",
            )
        ]
    return bounded[0], []


def _validate_endpoint_attestation(
    *,
    endpoint: SqlRiskScopeEndpoint,
    attestation: SqlRiskScopeEndpointAttestation,
    location: str,
    original_task: str,
) -> tuple[
    tuple[int, int] | None,
    tuple[int, int] | None,
    list[SqlRiskScopeExtractionIssue],
]:
    field_quote = attestation.field_name
    table_location, table_issues = _validate_component_attestation(
        component=endpoint.table_name,
        quote=attestation.table_name,
        location=f"{location}.table_name",
        original_task=original_task,
        right_component=field_quote,
    )
    if endpoint.field_name is None:
        return table_location, None, table_issues
    assert field_quote is not None
    field_location, field_issues = _validate_component_attestation(
        component=endpoint.field_name,
        quote=field_quote,
        location=f"{location}.field_name",
        original_task=original_task,
        left_component=attestation.table_name,
    )
    return table_location, field_location, [*table_issues, *field_issues]


def _validate_endpoints(
    contract: SqlRiskScopeExtraction,
    original_task: str,
) -> tuple[
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
    list[SqlRiskScopeExtractionIssue],
]:
    source_table_location, source_field_location, source_issues = (
        _validate_endpoint_attestation(
            endpoint=contract.source,
            attestation=contract.origin.source,
            location="origin.source",
            original_task=original_task,
        )
    )
    target_table_location, target_field_location, target_issues = (
        _validate_endpoint_attestation(
            endpoint=contract.target,
            attestation=contract.origin.target,
            location="origin.target",
            original_task=original_task,
        )
    )
    return (
        source_table_location,
        source_field_location,
        target_table_location,
        target_field_location,
        [*source_issues, *target_issues],
    )


def _validate_file_id_attestation(
    contract: SqlRiskScopeExtraction,
    original_task: str,
) -> tuple[tuple[int, int] | None, list[SqlRiskScopeExtractionIssue]]:
    if contract.file_id is None:
        return None, []

    quote = contract.origin.file_id
    assert quote is not None  # Enforced by the native schema model validator.
    if quote != quote.strip() or quote != str(contract.file_id):
        return None, [
            _issue(
                "file_id_mismatch",
                "origin.file_id",
                "attestation must be the exact selected integer token",
            )
        ]

    locations = _exact_locations(original_task, quote)
    if not locations:
        return None, [
            _issue(
                "file_id_not_found",
                "origin.file_id",
                "verbatim file_id attestation is absent from original_task",
            )
        ]
    bounded = _bounded_exact_locations(original_task, quote)
    if not bounded:
        return None, [
            _issue(
                "file_id_not_bounded",
                "origin.file_id",
                "attested file_id is only a partial numeric/token value",
            )
        ]
    return bounded[0], []


def _schema_issues(error: ValidationError) -> list[SqlRiskScopeExtractionIssue]:
    issues: list[SqlRiskScopeExtractionIssue] = []
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(item) for item in detail.get("loc", ()))
        issues.append(
            _issue(
                "schema_error",
                location or "native_call",
                str(detail.get("msg") or "invalid native-call schema"),
            )
        )
    return issues or [
        _issue("schema_error", "native_call", "invalid native-call schema")
    ]


def validate_sql_risk_scope_extraction(
    payload: SqlRiskScopeExtraction | Mapping[str, Any],
    *,
    original_task: str,
) -> SqlRiskScopeExtractionValidation:
    """Validate one model-owned native call and fail closed on any issue.

    The function never invokes a model and never changes the selected mode or
    endpoint.  On failure, ``contract`` is deliberately absent so a caller
    cannot accidentally continue with partial or repaired-by-code scope.
    """

    try:
        contract = (
            payload
            if isinstance(payload, SqlRiskScopeExtraction)
            else SqlRiskScopeExtraction.model_validate(payload)
        )
    except ValidationError as error:
        return SqlRiskScopeExtractionValidation(
            status="invalid",
            issues=_schema_issues(error),
        )

    task = str(original_task or "")
    (
        source_table_location,
        source_field_location,
        target_table_location,
        target_field_location,
        endpoint_issues,
    ) = _validate_endpoints(contract, task)
    file_location, file_issues = _validate_file_id_attestation(contract, task)
    issues = [*endpoint_issues, *file_issues]
    if issues:
        return SqlRiskScopeExtractionValidation(
            status="invalid",
            issues=issues,
        )

    assert source_table_location is not None
    assert target_table_location is not None
    return SqlRiskScopeExtractionValidation(
        status="valid",
        contract=contract,
        origin_locations=SqlRiskScopeOriginLocations(
            source_table_start=source_table_location[0],
            source_table_end=source_table_location[1],
            source_field_start=(
                source_field_location[0] if source_field_location else None
            ),
            source_field_end=(
                source_field_location[1] if source_field_location else None
            ),
            target_table_start=target_table_location[0],
            target_table_end=target_table_location[1],
            target_field_start=(
                target_field_location[0] if target_field_location else None
            ),
            target_field_end=(
                target_field_location[1] if target_field_location else None
            ),
            file_id_start=(file_location[0] if file_location else None),
            file_id_end=(file_location[1] if file_location else None),
        ),
    )


def render_sql_risk_scope_extraction_repair(
    issues: list[SqlRiskScopeExtractionIssue],
) -> str:
    """Render validation errors for the caller's sole optional repair call."""

    if not issues:
        raise ValueError("repair instructions require at least one issue")
    details = "\n".join(
        f"- [{issue.code}] {issue.location}: {issue.message}"
        for issue in issues
    )
    return (
        f"Предыдущий native call `{SQL_RISK_SCOPE_EXTRACTION_TOOL_NAME}` "
        "не прошёл fail-closed проверку. Это единственная разрешённая "
        "repair-попытка: не меняй intent и не придумывай literals; исправь "
        "схему/дословные attestations по ошибкам:\n"
        f"{details}"
    )


__all__ = [
    "MAX_SQL_RISK_SCOPE_EXTRACTION_ATTEMPTS",
    "SQL_RISK_SCOPE_EXTRACTION_PROMPT",
    "SQL_RISK_SCOPE_EXTRACTION_TOOL_NAME",
    "SQL_RISK_SCOPE_EXECUTION_MODES",
    "SqlRiskScopeEndpoint",
    "SqlRiskScopeEndpointAttestation",
    "SqlRiskScopeExecutionMode",
    "SqlRiskScopeExtraction",
    "SqlRiskScopeExtractionIssue",
    "SqlRiskScopeExtractionValidation",
    "SqlRiskScopeOriginAttestation",
    "SqlRiskScopeOriginLocations",
    "render_sql_risk_scope_extraction_repair",
    "validate_sql_risk_scope_extraction",
]
