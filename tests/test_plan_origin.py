import pytest

from agents.contracts import WorkerPlan
from agents.plan_origin import PlanOriginError, validate_worker_plan_origin


def _plan(*steps):
    return WorkerPlan.model_validate({"steps": list(steps)})


def test_plan_origin_accepts_literal_pair_and_explicit_file_id():
    plan = _plan(
        {
            "task": "Read exact src_np to tgt_np mapping for file_id=9101.",
            "scope": {"file_id": 9101},
        }
    )

    validate_worker_plan_origin(
        plan,
        "Для file_id=9101 проверь src_np → tgt_np.",
    )


def test_plan_origin_rejects_file_id_invented_before_filename_dependency():
    plan = _plan(
        {
            "task": "Разрешить synthetic.xlsx в file_id.",
            "scope": {"filename": "synthetic.xlsx"},
        },
        {
            "task": "Прочитать src_np → tgt_np для file_id=1.",
            "scope": {"file_id": 1},
            "dependencies": [1],
        },
    )

    with pytest.raises(PlanOriginError, match="file_id=1"):
        validate_worker_plan_origin(
            plan,
            "Для файла synthetic.xlsx проверь src_np → tgt_np.",
        )


def test_plan_origin_rejects_rewritten_literal_endpoint():
    plan = _plan(
        {
            "task": "Прочитать точный mapping source_np → target_np.",
            "entity": {"role": "unknown", "table": "source_np"},
        }
    )

    with pytest.raises(PlanOriginError, match="src_np"):
        validate_worker_plan_origin(
            plan,
            "Оцени nullable для src_np.id → tgt_np.id.",
        )


def test_plan_origin_rejects_global_listing_that_drops_exact_pair():
    plan = _plan(
        {
            "task": "Перечислить все таблицы и все "
            "сохранённые правила.",
            "entity": {"role": "unknown", "table": "s2t_transformations"},
            "coverage": "all_matches",
        }
    )

    with pytest.raises(PlanOriginError) as error:
        validate_worker_plan_origin(
            plan,
            "Оцени write semantics точной пары src_np → tgt_np.",
        )

    assert "src_np" in str(error.value)
    assert "tgt_np" in str(error.value)


def test_plan_origin_accepts_endpoint_split_into_table_and_field():
    plan = _plan(
        {
            "task": "Прочитать source metadata.",
            "entity": {"role": "source", "table": "src_np", "field": "id"},
        },
        {
            "task": "Прочитать target metadata.",
            "entity": {"role": "target", "table": "tgt_np", "field": "id"},
        },
    )

    validate_worker_plan_origin(
        plan,
        "Оцени nullable для src_np.id → tgt_np.id.",
    )


def test_plan_origin_rejects_field_level_pair_that_keeps_only_tables():
    plan = _plan(
        {
            "task": "Прочитать полный mapping src_np → tgt_np.",
            "entity": {"role": "source", "table": "src_np"},
        }
    )

    with pytest.raises(PlanOriginError, match="src_np.id"):
        validate_worker_plan_origin(
            plan,
            "Оцени value changes для src_np.id → tgt_np.id.",
        )


def test_plan_origin_rejects_invented_structured_file_or_sheet_scope():
    plan = _plan(
        {
            "task": "Прочитать mapping.",
            "scope": {"filename": "other.xlsx", "sheet_name": "S2T"},
        }
    )

    with pytest.raises(PlanOriginError) as error:
        validate_worker_plan_origin(
            plan,
            "Для файла expected.xlsx проверь mapping.",
        )

    assert "other.xlsx" in str(error.value)
    assert "S2T" in str(error.value)


def test_plan_origin_does_not_treat_natural_language_arrow_as_identifier():
    plan = _plan({"task": "Прочитать нужные факты."})

    validate_worker_plan_origin(
        plan,
        "Опиши переход источник → результат без "
        "технических имён.",
    )


def test_plan_origin_accepts_literals_from_coordinator_context():
    plan = _plan(
        {
            "task": "Прочитать src_np → tgt_np для file_id=7.",
            "scope": {"file_id": 7, "filename": "known.xlsx"},
        }
    )

    validate_worker_plan_origin(
        plan,
        "Проверь эту загрузку.",
        context="src_np → tgt_np; file_id=7; файл known.xlsx",
    )


def test_plan_origin_rejects_invented_structured_entity_literals():
    plan = _plan(
        {
            "task": "Прочитать объект.",
            "entity": {
                "role": "target",
                "table": "invented_target",
                "field": "invented_field",
            },
        }
    )

    with pytest.raises(PlanOriginError) as error:
        validate_worker_plan_origin(
            plan,
            "Покажи количество загруженных файлов.",
        )

    message = str(error.value)
    assert "step 1 entity.table='invented_target'" in message
    assert "step 1 entity.field='invented_field'" in message


def test_plan_origin_rejects_identifier_filter_but_allows_generic_filter_values():
    invented = _plan(
        {
            "task": "Прочитать строки.",
            "scope": {
                "filters": {
                    "target_table": "invented_target",
                    "nested": {"column_name": ["invented_field"]},
                    "file_id": 99,
                    "status": "active",
                    "limit": 25,
                    "include_nulls": False,
                }
            },
        }
    )

    with pytest.raises(PlanOriginError) as error:
        validate_worker_plan_origin(
            invented,
            "Прочитай активные строки.",
        )

    message = str(error.value)
    assert "scope.filters.target_table='invented_target'" in message
    assert "scope.filters.nested.column_name[0]='invented_field'" in message
    assert "scope.filters.file_id=99" in message
    assert "scope.filters.status" not in message
    assert "scope.filters.limit" not in message
    assert "scope.filters.include_nulls" not in message

    generic = _plan(
        {
            "task": "Прочитать строки.",
            "scope": {
                "filters": {
                    "status": "active",
                    "limit": 25,
                    "include_nulls": False,
                }
            },
        }
    )
    validate_worker_plan_origin(
        generic,
        "Прочитай активные строки.",
    )


def test_plan_origin_accepts_identifier_from_source_and_declared_dependency():
    plan = _plan(
        {
            "task": "Прочитать source metadata.",
            "entity": {
                "role": "source",
                "table": "src_orders",
                "field": "order_id",
            },
        },
        {
            "task": "Отфильтровать зависимый "
            "результат по той же колонке.",
            "entity": {
                "role": "target",
                "table": "{{step_1.target_table}}",
            },
            "scope": {
                "filters": {
                    "source_field": "order_id",
                    "target_table": {
                        "from_step": 1,
                        "field": "canonical_target_table",
                    },
                }
            },
            "dependencies": [1],
        },
    )

    validate_worker_plan_origin(
        plan,
        "Для src_orders.order_id найди и проверь target по "
        "сохранённому S2T.",
    )


def test_plan_origin_rejects_reference_to_undeclared_dependency():
    plan = _plan(
        {"task": "Получить исходный результат."},
        {
            "task": "Прочитать зависимый target.",
            "scope": {
                "filters": {
                    "target_table": {
                        "from_step": 1,
                        "field": "canonical_target_table",
                    }
                }
            },
            "dependencies": [],
        },
    )

    with pytest.raises(PlanOriginError, match="не указан в dependencies"):
        validate_worker_plan_origin(
            plan,
            "Найди target по бизнес-смыслу.",
        )
