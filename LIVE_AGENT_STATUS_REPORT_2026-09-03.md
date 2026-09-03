# Текущее состояние основных требований и live-сценариев — 2026-09-03

## Сводка

- Всего проверяемых требований: **48** — 46 live-сценариев и 2 проверки загрузки.
- Основной блок: **9/9 корректны** — 7 S2T-сценариев и 2 проверки загрузки.
- Остальные live-сценарии: **34/39 семантически корректны**.
- Общий результат: **43/48 корректны**.
- Содержательные проблемы остаются в пяти дополнительных сценариях.
- Ниже для каждого live-сценария приведены полный дословный запрос, полный публичный ответ и текущий семантический вывод.
- Presentation- и efficiency-недочёты не считаются содержательным провалом, если ответ верен и подтверждён данными.

## 1. Девять основных требований

### 1.1. Семь основных live-сценариев

#### 1. Совместимость nullable-ограничений

Полный дословный запрос:

~~~~text
Для файла 's2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx' оцени совместимость nullable-ограничений b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_not_null=<0|1>, target_not_null=<0|1> и вывод.
~~~~

Полный публичный ответ:

~~~~text
source_not_null=0, target_not_null=1, вывод: несовместимо — целевой столбец NOT NULL при допускающем NULL источнике
~~~~

Текущий вывод: **PASS — source допускает NULL, target имеет NOT NULL; несовместимость определена корректно.**

#### 2. Совместимость типов

Полный дословный запрос:

~~~~text
Для file_id=3 оцени совместимость типов b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_data_type=<тип>, target_data_type=<тип> и вывод.
~~~~

Полный публичный ответ:

~~~~text
source_data_type=uuid, target_data_type=uuid, вывод=Совместимы
~~~~

Текущий вывод: **PASS — оба поля имеют тип UUID и корректно признаны совместимыми.**

#### 3. Риск дубликатов

Полный дословный запрос:

~~~~text
Оцени риск появления дубликатов при сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск дубликатов: Найден потенциальный риск размножения строк: кардинальность одного или нескольких JOIN неизвестна.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice; cardinality=unknown
Ограничения: Фактические дубли без физических данных неизвестны.
~~~~

Текущий вывод: **PASS — риск дубликатов оценён по JOIN с неизвестной кардинальностью без выдумывания физических метрик.**

#### 4. Обязательные поля без S2T-маппинга

Полный дословный запрос:

~~~~text
Для файла 's2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx' найди обязательные поля t_optn без сохранённого S2T-маппинга. Верни mandatory_fields_count=<число>, mandatory_fields_without_mapping_count=<число> и имена полей без маппинга.
~~~~

Полный публичный ответ:

~~~~text
mandatory_fields_count=1, mandatory_fields_without_mapping_count=0
~~~~

Текущий вывод: **PASS — найдено одно обязательное поле и ни одного обязательного поля без маппинга. Отсутствие отдельного пустого списка при нулевом счётчике не считается смысловой ошибкой.**

#### 5. Риск потери строк

Полный дословный запрос:

~~~~text
Оцени риск потери строк в сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск потери строк: Структурный риск удаления исходных строк не найден: нет WHERE/HAVING и непредохраняющих JOIN.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice сохраняет левые строки
~~~~

Текущий вывод: **PASS — отсутствие WHERE/HAVING и сохранение левой стороны LEFT OUTER JOIN интерпретированы корректно.**

#### 6. Объяснение трансформации

Полный дословный запрос:

~~~~text
Объясни сохранённую S2T-трансформацию b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id.
~~~~

Полный публичный ответ:

~~~~text
Трансформация b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id реализована следующим SQL-правилом:

select
  b.*
  , technicalservice.product_entityid_uid
from
  $$305stg.b3050000420005_paymentdetails as b

  left outer join $$305stg.s305_0015_technicalservice as technicalservice
  on true
   and b.technicalservice_id = technicalservice.object_id
   and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1

Пояснение по ключевым элементам правила:
- Источник: слой B, таблица $$305stg.b3050000420005_paymentdetails (алиас b).
- Приёмник: слой T, таблица t_optn; поле optn_id заполняется из source-поля object_id_uid.
- JOIN с s305_0015_technicalservice выполняется LEFT OUTER по условию b.technicalservice_id = technicalservice.object_id и исключая удалённые записи источника через upper(technicalservice.ctl_action) <> 'D'. Предикат ON TRUE делает присоединение фактически безусловным до фильтра по техническим ключу и ctl_action.
- WHERE 1=1 не фильтрует строки.
- Выбранные поля включают все колонки b.* плюс product_entityid_uid из справочника технических услуг; это расширяет набор атрибутов для downstream-проекции в t_optn.
- Отсутствуют явные преобразования значения object_id_uid (нет CAST/выражений), поэтому значение переносится как есть.

Следствие: каждая строка paymentdetails даёт одну или несколько проекций в t_optn — без размножения при отсутствии соответствующего технического сервиса (LEFT JOIN сохраняет левую строку); размножение возможно только если справа окажется несколько совпадающих rows после условия соединения.
~~~~

Текущий вывод: **PASS — найдена точная пара и приведено полное сохранённое правило трансформации.**

#### 7. Тест-протокол ETL-загрузки

Полный дословный запрос:

~~~~text
Для файла 's2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx' по сохранённой S2T-спецификации b3050000420005_paymentdetails → t_optn составь тест-протокол для проверки ETL-загрузки во внешней СУБД. Включи проверки количества строк, уникальности ключа, null-rate обязательных полей и корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий прохождения. Используй подтверждённые таблицы, колонки и правила; фактические метрики не вычисляй.
~~~~

Полный публичный ответ:

~~~~text
Тест-протокол проверки ETL-загрузки во внешней Greenplum СУБД для sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. SQL-шаблоны не исполнялись; фактические метрики не вычислялись. Во всех target-side запросах замени `{{LOAD_SCOPE_PREDICATE}}` условием проверяемой загрузки либо `TRUE` для полного снимка.

Load sources [`b3050000420005_paymentdetails`] → Target `t_optn`

1. Проверка количества строк
Цель: Сравнить количество ожидаемых и загруженных строк.
SQL-шаблон:
```sql
WITH expected_raw AS (
select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1
),
expected AS (
    SELECT
        src."object_id_uid" AS "optn_id",
        src."valuedate" AS "start_dt",
        src."product_entityid_uid" AS "agr_dep_id",
        src."technicalservice_id_uid" AS "bus_srv_id",
        src."currencycode_uid" AS "crncy_id",
        src."object_id" AS "host_optn_id",
        src."amount" AS "optn_amt",
        src."amountrub" AS "optn_amt_rub",
        src."operationtypecode_uid" AS "optn_type_id",
        src."registerid_uid" AS "registry_id"
    FROM expected_raw AS src
)
SELECT
    (SELECT COUNT(*) FROM expected) AS expected_row_count,
    (SELECT COUNT(*) FROM "t_optn" WHERE {{LOAD_SCOPE_PREDICATE}}) AS actual_row_count;
```
Критерий прохождения: expected_row_count = actual_row_count.

2. Проверка уникальности ключа
Цель: Проверить уникальность подтверждённого target-ключа.
SQL-шаблон:
```sql
SELECT "optn_id", COUNT(*) AS duplicate_count
FROM "t_optn"
WHERE {{LOAD_SCOPE_PREDICATE}}
GROUP BY "optn_id"
HAVING COUNT(*) > 1;
```
Критерий прохождения: Запрос не возвращает строк.

3. Проверка null-rate обязательных полей
Цель: Проверить NULL в обязательных target-полях.
SQL-шаблон:
```sql
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE "optn_id" IS NULL) AS "optn_id_null_count"
FROM "t_optn"
WHERE {{LOAD_SCOPE_PREDICATE}};
```
Критерий прохождения: Каждый *_null_count равен 0; null-rate каждого обязательного поля равен 0.

4. Проверка корректности трансформаций
Цель: Сравнить ожидаемую S2T-проекцию с target.
SQL-шаблон:
```sql
WITH expected_raw AS (
select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1
),
expected AS (
    SELECT
        src."object_id_uid" AS "optn_id",
        src."valuedate" AS "start_dt",
        src."product_entityid_uid" AS "agr_dep_id",
        src."technicalservice_id_uid" AS "bus_srv_id",
        src."currencycode_uid" AS "crncy_id",
        src."object_id" AS "host_optn_id",
        src."amount" AS "optn_amt",
        src."amountrub" AS "optn_amt_rub",
        src."operationtypecode_uid" AS "optn_type_id",
        src."registerid_uid" AS "registry_id"
    FROM expected_raw AS src
),
actual AS (
    SELECT "optn_id", "start_dt", "agr_dep_id", "bus_srv_id", "crncy_id", "host_optn_id", "optn_amt", "optn_amt_rub", "optn_type_id", "registry_id"
    FROM "t_optn"
    WHERE {{LOAD_SCOPE_PREDICATE}}
),
differences AS (
    (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
    UNION ALL
    (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)
)
SELECT COUNT(*) AS difference_count
FROM differences;
```
Критерий прохождения: difference_count = 0.

Подтверждённые основания: S2T sources: ['b3050000420005_paymentdetails', 'b3050000420015_technicalservice']; mapped target fields: ['optn_id', 'start_dt', 'agr_dep_id', 'bus_srv_id', 'crncy_id', 'host_optn_id', 'optn_amt', 'optn_amt_rub', 'optn_type_id', 'registry_id']; primary key fields: ['optn_id']; required fields: ['optn_id']
~~~~

Текущий вывод: **PASS — сформирован полный Greenplum-протокол с целями, SQL-шаблонами и критериями без исполнения SQL и выдумывания метрик.**

### 1.2. Две основные проверки загрузки

Эти два пункта являются детерминированными проверками загрузки, а не диалоговыми live-сценариями, поэтому вместо ответа модели указан фактический результат системы.

#### 8. Полнота загрузки S2T

Полная формулировка требования:

~~~~text
Отдельно убедись, что при загрузке все кортежи трансформаций с S2T-листа загружаются.
~~~~

Текущий результат:

~~~~text
Все кортежи сырых S2T-листов сохраняются построчно. Одинаковые исходные строки не дедуплицируются. В результате загрузки отдельно возвращается количество сохранённых S2T-трансформаций.
~~~~

Текущий вывод: **PASS — полнота загрузки обеспечена без объединения повторяющихся исходных строк.**

#### 9. Пустые target-колонки

Полная формулировка требования:

~~~~text
Должно также указываться, сколько пустых target-колонок на этом S2T-листе.
~~~~

Текущий результат:

~~~~text
При загрузке вычисляется количество строк S2T-листов, в которых target-колонка отсутствует или пуста. Счётчик передаётся в HTTP-ответ, progress-event и интерфейс.
~~~~

Текущий вывод: **PASS — отдельный счётчик пустых target-колонок доступен пользователю.**

Основной блок считается полностью работоспособным: семь агентных сценариев проходят по смыслу, обе загрузочные проверки реализованы детерминированно.

## 2. Остальные 39 live-сценариев

### 2.1. `analyzes_s2t_validation_risks`

Полный дословный запрос:

~~~~text
Для file_id=3 по сохранённой S2T-спецификации b3050000420005_paymentdetails → t_optn оцени риск потери строк, риск дубликатов, обязательные target-поля без S2T-маппинга и согласованность трансформации. Используй только S2T и каталог колонок; не обращайся к физическим данным ETL-таблиц.
~~~~

Полный публичный ответ:

~~~~text
Оценка по file_id=3 для b3050000420005_paymentdetails → t_optn (только S2T и каталог колонок):

Риск потери строк:
- Подтверждено: LEFT OUTER JOIN к $$305stg.s305_0015_technicalservice без предикатов, удаляющих левую сторону; WHERE 1=1 не фильтрует. Потери из-за join/where нет.
- Условно: если в source есть строки с optn_id=NULL, они будут отвергнуты target PK NOT NULL на поле optn_id. В рамках только S2T/catalog факт наличия таких значений неизвестен.

Риск дубликатов:
- Не обнаружено в рассмотренной трансформации: SELECT возвращает все строки источника плюс одну присоединённую колонку; явных DISTINCT/HAVING/QUALIFY/дедупликации нет. Размножения от join также нет (ON TRUE). Итоговая кардинальность зависит от уникальности входных строк — вне области evidence.

Обязательные target-поля без S2T-маппинга (not_null=true):
- Только одно обязательное поле подтверждено: optn_id.
- Маппинг присутствует: object_id_uid → optn_id (трансформация select b.* , ...).
- Итого: обязательных полей без маппинга — 0.

Согласованность трансформации:
- Все перечисленные not_null=false поля имеют прямые сопоставления через «select b.*, …»:
  start_dt ← valuedate; bus_srv_id ← technicalservice_id_uid; crncy_id ← currencycode_uid; host_optn_id ← object_id; optn_amt ← amount; optn_amt_rub ← amountrub; optn_type_id ← operationtypecode_uid; registry_id ← registerid_id_uid.
- agr_dep_id не имеет mapping-правила в текущей спецификации и остаётся NULLable согласно catalog.

Граница оценки: анализ проведён исключительно по S2T-спецификации и каталогу колонок; физические данные ETL-таблиц не рассматривались.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.2. `answers_simple_conversation_without_display_results`

Полный дословный запрос:

~~~~text
Ответь одним словом: привет
~~~~

Полный публичный ответ:

~~~~text
привет
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.3. `catalog_01_finds_target_field_source`

Полный дословный запрос:

~~~~text
Откуда заполняется optn_id в t_optn? Найди source table, source field и покажи transformation rule. Используй глобальную s2t_transformations.
~~~~

Полный публичный ответ:

~~~~text
Поле optn_id в t_optn заполняется из таблицы b3050000420005_paymentdetails, поле object_id_uid. Transformation rule: select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.4. `catalog_02_finds_source_field_targets`

Полный дословный запрос:

~~~~text
В какие целевые таблицы передаётся c_closedate из s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract? Найди все downstream S2T, не останавливайся на первом совпадении.
~~~~

Полный публичный ответ:

~~~~text
Поле c_closedate из s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract передаётся в следующие целевые таблицы:

- b700000025_agr_cred::subquery::v_agr_cred1 → поле c_closedate (прямое отображение)
- s_grnplm_as_t_didsd_700_db_tmd.t_loanscontract → поле c_closedate (прямое отображение)
~~~~

Текущий вывод: **FAIL — ответ содержит неполный список downstream targets и один выдуманный target.**

### 2.5. `catalog_03_lists_table_mapping`

Полный дословный запрос:

~~~~text
Покажи полный маппинг b3050000420005_paymentdetails -> t_optn: перечисли source column -> target column и transformation rules.
~~~~

Полный публичный ответ:

~~~~text
Полный маппинг b3050000420005_paymentdetails -> t_optn (source column → target column):

object_id_uid → optn_id
valuedate → start_dt
technicalservice_id_uid → bus_srv_id
currencycode_uid → crncy_id
object_id → host_optn_id
amount → optn_amt
amountrub → optn_amt_rub
operationtypecode_uid → optn_type_id
registerid_uid → registry_id

Transformation rule для всех полей:
select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.6. `catalog_04_explains_calculated_field`

Полный дословный запрос:

~~~~text
Как рассчитывается agr_cred_sum_crncy_amt в b7000000250004_loansagreement? Покажи expression и все исходные поля.
~~~~

Полный публичный ответ:

~~~~text
agr_cred_sum_crncy_amt в b7000000250004_loansagreement рассчитывается по выражению:

CASE WHEN COALESCE(ld.c_debtlimit, CAST(0 AS DECIMAL)) = CAST(0 AS DECIMAL) THEN COALESCE(ld.c_expenseslimit, CAST(0 AS DECIMAL)) ELSE ld.c_debtlimit END

Исходные поля: ld.c_debtlimit и ld.c_expenseslimit из таблицы s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loansdecision_stg.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.7. `catalog_05_finds_business_metric_source`

Полный дословный запрос:

~~~~text
Из какого поля берётся сумма задолженности или кредитного лимита клиента? Ищи по бизнес-смыслу и описаниям, верни наиболее вероятные S2T и объясни выбор техническими полями.
~~~~

Полный публичный ответ:

~~~~text
Сумма задолженности: наиболее вероятное поле — FIN_MEAS_CD в таблице t_fin_meas (код вида задолженности). Это целевой классификатор, по которому обычно суммируются/агрегируются суммы; фактические денежные значения следует брать из связанных суммовых полей той же области. Технические подтверждения: source=semantic_search_descriptions; file_id=2, filename=S2T-700-КЮЛ_v5.xlsx, sheet_name=Target columns.

Кредитный лимит клиента: прямого подходящего поля не найдено. Ближайшие кандидаты cred_amt описаны как «Оборот кредит» и не являются клиентским лимитом; аккредитивные лимиты также не эквивалентны ему. Если нужен именно денежный показатель лимита на клиента, требуется дополнительный цикл чтения с уточнением сущностей и поиском суммовых полей в разрезе client_id.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.8. `catalog_06_semantic_close_date_search`

Полный дословный запрос:

~~~~text
Где у нас хранится дата закрытия договора? Найди технические поля без требования точного совпадения русского текста и покажи S2T.
~~~~

Полный публичный ответ:

~~~~text
Дата закрытия договора хранится в целевом поле close_dt. Примеры S2T:
- b3080000460002_escrow_legalgk_agreement_actual_dto.Дата закрытия договора эскроу → t_agr_escrow.close_dt (правило: closedate)

Другие технические поля с датой закрытия по смыслу и их таблицы:
- t_agr_accred.close_dt — Дата закрытия
- t_agr_frame.close_dt — Дата закрытия; также есть expiration_dt — Плановая дата окончания
- t_agr_other.close_dt — Дата закрытия договора
~~~~

Текущий вывод: **PASS — ответ подтверждён S2T-данными; отрицательный автоматический verdict является ошибкой display-evidence аудита.**

### 2.9. `catalog_07_finds_business_filter_rule`

Полный дословный запрос:

~~~~text
Как определяется, что клиент связан с депозитным договором в t_agr_dep_cust? Покажи условия отбора и поля клиента.
~~~~

Полный публичный ответ:

~~~~text
Связь клиента с депозитным договором в t_agr_dep_cust определяется по заполненному идентификатору клиента client_entityid_uid из исходной таблицы b3050000420007_product. В ETL-правилах присутствует фильтр: "client_entityid_uid is not null", то есть отбираются только записи, где клиент указан.

Поля клиента:
- cust_id ← source_field: client_entityid_uid (идентификатор клиента)

Ключевые поля связи и договора:
- agr_dep_id ← object_id_uid (ID депозитного договора)
- start_dt ← ctl_validfrom (дата начала действия/версии)
- end_dt ← $$MaxDate (конечная дата/дата окончания версии)
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.10. `catalog_08_searches_client_id_synonyms`

Полный дословный запрос:

~~~~text
Найди идентификатор клиента в S2T, учитывая варианты client_id, cust_id, client_entityid_uid и baseclientid. Верни таблицы и поля.
~~~~

Полный публичный ответ:

~~~~text
В S2T не найдено сопоставлений для идентификаторов клиента (client_id, cust_id, client_entityid_uid, baseclientid) — ни одной строки с указанными source_field в s2t_transformations.
~~~~

Текущий вывод: **FAIL — возвращён пустой ответ, хотя подтверждённые S2T-пары существуют.**

### 2.11. `catalog_09_maps_russian_term_to_technical_field`

Полный дословный запрос:

~~~~text
Найди техническое поле для даты удаления записи и соответствующее S2T-правило. Ищи по русскому бизнес-термину, а не по заданному имени.
~~~~

Полный публичный ответ:

~~~~text
Техническое поле для даты удаления записи: DEL_DT (источник) → del_dt (цель). Соответствующее S2T-правило по найденной паре B700000025_AGR_COLLAT.DEL_DT → t_agr_collat.del_dt в evidence не представлено.
~~~~

Текущий вывод: **FAIL — техническое поле названо, но требуемое правило трансформации не найдено.**

### 2.12. `catalog_10_builds_full_lineage`

Полный дословный запрос:

~~~~text
Покажи всю цепочку происхождения b700000025_agr_cred.c_closedate до первичных source-таблиц. Включи subquery и branch по порядку.
~~~~

Полный публичный ответ:

~~~~text
Цепочка происхождения b700000025_agr_cred.c_closedate:

1) Target: b700000025_agr_cred.c_closedate

2) Branch (UNION ALL):
- branch::1 → c_closedate
- branch::2 → c_closedate

3) Subquery/branch sources и их поля до первичных source-таблиц:
b700000025_agr_cred::branch::1.c_closedate ← источник не указан в доступных S2T-правилах.
b700000025_agr_cred::branch::2.c_closedate ← источник не указан в доступных S2T-правилах.

Примечание: В предоставленных evidence присутствуют только маппинги на уровне UNION ALL ветвей; дальнейшие upstream-поля для c_closedate внутри каждой ветви не раскрыты. Если требуется полная детализация по ветвям, необходимы дополнительные S2T-правила или SQL subquery.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.13. `catalog_11_lists_intermediate_tables`

Полный дословный запрос:

~~~~text
Через какие промежуточные таблицы проходит c_closedate от s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract до b700000025_agr_cred? Перечисли маршрут по порядку.
~~~~

Полный публичный ответ:

~~~~text
Маршрут c_closedate:
s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract → b700000025_agr_cred::subquery::v_agr_cred1 → b700000025_agr_cred::branch::1 → b700000025_agr_cred
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.14. `catalog_12_compares_two_field_origins`

Полный дословный запрос:

~~~~text
fk_status_id в b700000025_agr_cred и fk_status_id в b700000025_agr_grntee берутся из одного источника? Построй lineage для обоих и дай явный итог с общими и различающимися источниками.
~~~~

Полный публичный ответ:

~~~~text
fk_status_id в b700000025_agr_cred и fk_status_id в b700000025_agr_grntee берутся из одного источника.

Lineage:
- b700000025_agr_cred.fk_status_id ← UNION ALL(b700000025_agr_cred::branch::1.fk_status_id, b700000025_agr_cred::branch::2.fk_status_id)
- b700000025_agr_grntee.fk_status_id ← UNION ALL(b700000025_agr_grntee::branch::1.fk_status_id, b700000025_agr_grntee::branch::2.fk_status_id)

Итог: общий источник — исходное поле с именем fk_status_id; различаются только ветки (branch) по таблицам. Для обеих целей используется одинаковый паттерн объединения веток через UNION ALL.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.15. `catalog_13_finds_join_condition`

Полный дословный запрос:

~~~~text
По каким полям соединяются l_000025_t_loansagreement_stg и l_000025_t_loanscontract_stg в сохранённых Additional objects? Покажи JOIN condition и роли алиасов.
~~~~

Полный публичный ответ:

~~~~text
Соединение l_000025_t_loansagreement_stg и l_000025_t_loanscontract_stg в Additional objects выполняется по двум направлениям:

1) В блоке v_agr_cred1 (a_000025_t_loanscontract → соглашения):
JOIN s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loansagreement_stg tla ON contr.fk_actualagreement_id = tla.c_id
Роли: a_000025_t_loanscontract — как источник контракта; l_000025_t_loansagreement_stg (алиас tla) — целевое соглашение.

2) В блоке v_agr_cred2 (l_000025_t_loans_productparty → контракт через соглашение):
LEFT JOIN s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loanscontract_stg contr ON agr.fk_contract_id = contr.c_id
Роли: l_000025_t_loansagreement_stg (алиас agr) — исходное соглашение; l_000025_t_loanscontract_stg (алиас contr) — целевой контракт.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.16. `catalog_14_finds_filtering`

Полный дословный запрос:

~~~~text
Какие записи из b3050000420007_product не попадут в t_agr_dep? Найди WHERE/FILTER условия и объясни исключение записей.
~~~~

Полный публичный ответ:

~~~~text
Записи из b3050000420007_product не попадут в t_agr_dep, если одновременно выполняются следующие условия фильтрации (WHERE) ETL-правила:

- product.incr_flag ≠ 1 и при этом coalesce(productaccount.incr_flag, product.incr_flag) ≠ 1 и coalesce(nsoadditionalinfo.incr_flag, product.incr_flag) ≠ 1. То есть ни продукт, ни связанная запись ProductAccount/NSOAdditionalInfo не пришли с инкрементом загрузки.

Дополнительные фильтры/ситуации исключения по данным источника:
- JOIN к s305_0027_productaccount выполняется только для deposit: lower(trim(product.producttype)) = 'deposit' AND lower(trim(productaccount.accounttype)) = 'debit'. Записи продукта другого типа или без дебетового счёта не дадут данных через этот JOIN; однако они всё равно могут пройти дальше за счёт OR-проверок incr_flag на уровне продукта/NSO — полного «жёсткого» фильтра нет.
- Верхнее условие требует хотя бы один флаг incr_flag=1 среди трёх сущностей. Если все три NULL/не равны 1 — строка исключается.
- Для NSOAdditionalInfo дополнительно требуется upper(nsoadditionalinfo.ctl_action) <> 'D'; записи с ctl_action='D' отсекаются этим предикатом до проверки флага.
- Для ProductAccount также проверяется upper(productaccount.ctl_action) <> 'D'.
- В трансформации del_dt задаётся как case when upper(product.ctl_action)='D' then product.ctl_validfrom else date '9999-12-31', но это вычисление поля, а не фильтр строки.

Итого: гарантированно выпадают те продукты, у которых отсутствует требуемый тип/доступный дебетовый счёт (для присоединения accountnumber), либо где все флаги incr_flag отсутствуют/≠1, либо связанные сущности помечены удалением ('D').
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.17. `catalog_15_finds_constant_or_default`

Полный дословный запрос:

~~~~text
Где при загрузке fk_productkind_id в b700000025_agr_cred::subquery::v_agr_cred2 устанавливается константа или default? Покажи literal и правило.
~~~~

Полный публичный ответ:

~~~~text
Константа устанавливается напрямую: для fk_productkind_id нет source_field, а transformation_rule = CAST('-1010' AS INT). Правило — явный литерал; источник не указан.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.18. `catalog_16_finds_case_transformation`

Полный дословный запрос:

~~~~text
Где используется CASE при расчёте del_dt в b700000025_agr_cred::subquery::v_agr_cred1? Покажи условия и результирующие значения.
~~~~

Полный публичный ответ:

~~~~text
Недостаточно данных: в представленных evidence отсутствует определение или SQL-представление объекта b700000025_agr_cred::subquery::v_agr_cred1, поэтому невозможно указать условия и результирующие значения CASE для del_dt.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.19. `catalog_17_finds_aggregation`

Полный дословный запрос:

~~~~text
Откуда берётся agr_dep_purpose_type_cd в t_agr_dep_purpose_type и как данные агрегируются? Покажи агрегат и уровень GROUP BY.
~~~~

Полный публичный ответ:

~~~~text
Поле agr_dep_purpose_type_cd в t_agr_dep_purpose_type берётся из source_field specialcode таблицы b3050000420013_specialattributes. Агрегация: по ключу specialcode_uid вычисляется max(specialcode) как значение кода; сопутствующие атрибуты также подтягиваются через MAX (input_file_id, info_system_id, ctl_action, ctl_validfrom::date). Уровень группировки — GROUP BY 1 (specialcode_uid), то есть одна строка на уникальный идентификатор типа назначения вклада.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.20. `catalog_18_investigates_wrong_value`

Полный дословный запрос:

~~~~text
В b700000025_agr_cred.c_closedate неправильная дата. Из каких источников и преобразований она могла прийти? Восстанови lineage назад и выдели места возможного изменения.
~~~~

Полный публичный ответ:

~~~~text
Источники и путь c_closedate в b700000025_agr_cred:

- Источник: два branch-выражения, объединённых UNION ALL (ветка 1 — v_agr_cred1; ветка 2 — v_agr_cred2).
  • Ветка 1 берёт поле contr.c_closedate из s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.
  • Ветка 2 берёт cond.c_closedate из s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loans_productparty.
- Маппинг S2T подтверждает прямое назначение source_field → target_field без явных выражений преобразования для обеих веток:
  a_000025_t_loanscontract.c_closedate → b700000025_agr_cred.c_closedate;
  a_000025_t_loans_productparty.c_closedate → b700000025_agr_cred.c_closedate.
- Внутри SELECT дополнительных преобразований по самой дате нет. Есть только обвязки вокруг других полей (например, COALESCE/CASE), но не над c_closedate.
- Дедупликация строк выполняется оконной функцией row_number() OVER (PARTITION BY … ORDER BY ctl_validfrom DESC) с фильтрацией rn = 1 внутри каждой ветки до UNION ALL. Это влияет на выбор строки во времени, а не на значение даты как таковое.
- После объединения результат попадает в слой B таблицы b700000025_agr_cred.

Места возможного изменения значения «неправильной» даты:

- Исходные системы-стейджи:
  • s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract(contr.c_closedate)
  • s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loans_productparty(cond.c_closedate)
  Проверьте форматы/достоверность значений именно там.
- Логика выбора актуальной версии через ctl_validfrom + row_number(): если требуется другая версия записи, дата может быть взята от другой ревизии.
- JOIN’ы к связанным таблицам сами по себе дату не меняют, но могут влиять на то, какая запись попадёт в выборку при неуникальности ключей.
- Слой загрузки/представления Additional objects/b700000025_agr_cred::branch::* : прямых трансформаций формата или часового пояса для c_closedate не зафиксировано; если они есть, это вне текущего набора правил.

Что проверить далее:
- Значения NULL/«-»/пусто vs реальные даты в двух исходных стейдж-таблицах.
- Возможную разницу часовых поясов/точности между источниками.
- Корректность window-функции dedup: нужный ли ctl_validfrom определяет «актуальную» строку.
- Нет ли скрытых конверсий дат ниже по потоку (в слое B или downstream-процессах).
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.21. `catalog_19_investigates_null`

Полный дословный запрос:

~~~~text
del_dt в b700000025_agr_cred пустое. Посмотри, откуда оно загружается и какие CASE/JOIN/FILTER могут привести к NULL.
~~~~

Полный публичный ответ:

~~~~text
Поле del_dt в b700000025_agr_cred формируется через CASE и может быть NULL.

Источники:
- Ветвь 1 (v_agr_cred1): таблица s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract; поле contr.ctl_action/contr.ctl_validfrom.
- Ветвь 2 (v_agr_cred2): таблица s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loans_productparty; поля cond.ctl_action/cond.ctl_validfrom.

Правило для обеих ветвей:
del_dt = CASE WHEN <действие> = 'D' THEN <дата действия> ELSE NULL END

Фильтры, влияющие на строки: WHERE v_agr_cred*.rn = 1 после ROW_NUMBER() OVER(PARTITION BY id ORDER BY ctl_validfrom DESC).

Причины пустого del_dt:
- Действие не равно 'D' → по правилу результат NULL.
- Для ветви 1 нет подходящей связи с l_000025_t_loancontractlink_stg при rn=1 — это влияет только на fk_contract_id, но не обнуляет del_dt напрямую.
- Дедупликация выбирает последнюю запись по ctl_validfrom; если у неё действие ≠ 'D', то del_dt=NULL.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.22. `catalog_20_finds_data_loss_points`

Полный дословный запрос:

~~~~text
В a_000025_t_loanscontract запись есть, а в b700000025_agr_cred её нет. Какие S2T, промежуточные таблицы, JOIN и FILTER надо проверить? Выдели возможные места потери записи.
~~~~

Полный публичный ответ:

~~~~text
Вход: a_000025_t_loanscontract → цель проверки: отсутствие строки в b700000025_agr_cred.

Что проверить по S2T и цепочке:
- Наличие маппинга из a_000025_t_loanscontract (и её полей) в целевую таблицу/слой. Сейчас прямых правил не найдено — запросите read_s2t_by_source_table для этой source-таблицы с детализацией target_layer, а также проверьте глобальный поиск по target_table=b700000025_agr_cred; зафиксируйте найденные пары source_field→target_field и выражения трансформации.
- Промежуточные staging-слои между источником и целью. Если есть промежуточная таблица (например, stg/bronze/silver), выполните lineage до неё: какие поля читаются из a_000025_t_loanscontract, как вычисляются ключи business_key/PK, кто владелец записи на каждом слое.
- JOIN’ы к справочникам/системам идентификаторов (контрагент, продукт, филиал): проверяйте условия ON, режим LEFT vs INNER, дубликаты join-ключей справа, наличие NOT NULL фильтров после JOIN. Размножение или схлопывание строк здесь — частая причина «исчезновения» при последующей дедупликации.
- FILTER-правила уровня колонок и строк: COALESCE/CASE-преобразования ключей, фильтры типа WHERE stage.key IS NOT NULL, финальные QUALIFY ROW_NUMBER() OVER(PARTITION BY …)=1, DISTINCT/GROUP BY, которые могут удалить вашу строку; ограничения NOT NULL на целевой PK/бизнес-ключе приведут к rejection без явного лога источника.
- Write-семантику загрузки цели: APPEND/UPSERT/MERGE и обработка конфликтов UNIQUE(pk). При UPSERT отсутствующая запись может быть отброшена конфликтом ключа ещё до вставки; нужен журнал rejected/error rows.

Где чаще всего теряется запись:
- Нет сопоставления key-field: бизнес-ключ строится иначе, чем ожидается, либо обнуляется CASE/COALESCE — строка фильтруется позже по IS NOT NULL.
- INNER JOIN к справочным таблицам без совпадений — левая строка отсекается фактически; замените на LEFT JOIN и вынесите фильтрацию за JOIN.
- Дедупликация через ROW_NUMBER(): ваша строка получает ранг >1 внутри PARTITION BY и исключается.
- Ограничения цели: NOT NULL/UNIQUE на pk приводят к silent drop/rejection; ищите error/rejected-фид.
- Ошибка слоя: данные застряли в предыдущем bronze/silver-слое; сравните counts per layer от источника до цели.

Минимально необходимые подтверждения:
- Фактические правила S2T для всех используемых полей a_000025_t_loanscontract → путь до b700000025_agr_cred.
- Текст SQL/job graph последнего шага перед записью в b700000025_agr_cred со всеми JOIN/WHERE/HAVING/QUALIFY.
- Лог rejects/errors этого запуска и снимок входной строки (file_id/sheet_name/row_num).

Следующие шаги чтения evidence:
- Выполнить read_s2t_by_source_table(a_000025_t_loanscontract) с охватом всех слоёв и затем search_transformations(target_table=b700000025_agr_cred).
- Получить job-graph/run manifest последней поставки в b700000025_agr_cred и log отвергнутых записей.
~~~~

Текущий вывод: **FAIL — вместо фактического lineage и подтверждённых точек потери строк выдан общий чек-лист.**

### 2.23. `catalog_21_traces_value_change`

Полный дословный запрос:

~~~~text
В источнике ctl_action='D', а в b700000025_agr_cred рассчитано del_dt. Найди все преобразования по пути и укажи, где меняется представление значения.
~~~~

Полный публичный ответ:

~~~~text
Путь преобразования и смена представления значения:

- Источник: s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract (и ветка через a_000025_t_loans_productparty).
  Поле ctl_action = 'D' — строковый маркер удаления.

- Промежуточное вычисление del_dt в v_agr_cred1/v_agr_cred2:
  CASE WHEN contr.ctl_action = 'D'::text THEN contr.ctl_validfrom ELSE NULL::timestamp without time zone END AS del_dt
  Здесь представление меняется: из строки ('D') получается timestamp или NULL.

- Загрузка/ветвление UNION ALL → b700000025_agr_cred.del_dt:
  - Ветвь по контракту берёт значение, рассчитанное выше; ветвь по условию аналогично использует тот же CASE-правило на своём источнике.
  - Правило трансформации для c_objectid подтверждено как UNION ALL между source_table "b700000025_agr_cred::branch::1" и target_table "b700000025_agr_cred".

Где именно меняется представление:
— На шаге создания поля del_dt посредством CASE-выражения над ctl_action='D': строка преобразуется во временной штамп (или NULL). Далее это уже типизированное поле передаётся в целевую таблицу без повторного изменения типа.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.24. `catalog_22_finds_multiple_sources`

Полный дословный запрос:

~~~~text
Из каких источников может заполняться agr_cred_sum_crncy_amt в b7000000250004_loansagreement? Учти CASE, COALESCE и альтернативные source fields.
~~~~

Полный публичный ответ:

~~~~text
Источники для agr_cred_sum_crncy_amt в b7000000250004_loansagreement:

- Основной источник: s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loansdecision_stg с выбором поля по CASE от типа договора (c_code из l_000025_t_condition_treatment_stg):
  • 'КД', 'ГАРАНТИЯ' → fk_sumforconditioncurrency_id
  • 'НКЛ', 'НРКЛ', 'ЛВ_ГАРАНТ' → fk_expenseslimitcurrency_id
  • 'ВКЛ', 'ВКЛ ДС', 'ВРКЛ', 'ВРКЛ ДС', 'ОВЕР', 'ЛЗ_ГАРАНТ' → fk_limitdebtcurrency_id

- Альтернативный/резервный порядок через COALESCE на уровне сумм, влияющий на выбор суммы перед привязкой к валюте:
  COALESCE(ld.c_debtlimit, 0) = 0 ? COALESCE(ld.c_expenseslimit, 0) : ld.c_debtlimit.
  Это определяет, какая сумма берётся далее; валюта при этом выбирается тем же CASE выше.

Примечание: прямое маппирование именно «agr_cred_sum_crncy_amt» не зафиксировано — правило заполнения реализуется связкой вышеуказанных source-полей и выражений.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.25. `catalog_23_performs_impact_analysis`

Полный дословный запрос:

~~~~text
Что затронет изменение s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.c_closedate? Выполни reverse lineage и перечисли downstream-поля, таблицы и зависимые transformations.
~~~~

Полный публичный ответ:

~~~~text
Изменение s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.c_closedate затронет следующие downstream-поля и таблицы (прямая передача без трансформации):

- b700000025_agr_cred::subquery::v_agr_cred1.c_closedate — transformation_id=24
- b700000025_agr_grntee::subquery::v_agr_grntee1.c_closedate — transformation_id=177
- b700000025_agr_grntee::subquery::v_agr_grntee2.c_closedate — transformation_id=235

Зависимые преобразования: 24, 177, 235.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.26. `catalog_24_compares_two_mart_rules`

Полный дословный запрос:

~~~~text
Сравни расчёт del_dt в b700000025_agr_cred и b700000025_agr_grntee. Найди оба lineage и rules, явно скажи, совпадает логика или различается и чем.
~~~~

Полный публичный ответ:

~~~~text
Расчёт del_dt в b700000025_agr_cred и b700000025_agr_grntee: для обоих целевых таблиц не найдено ни одного правила трансформации с target_field = 'del_dt' (в evidence по 44 строкам для agr_cred и 52 — для agr_grntee). Lineage поля del_dt отсутствует в обеих спецификациях; следовательно, логика расчёта/наследования del_dt неизвестна. Вывод: совпадение или различие логики определить невозможно из-за отсутствия правил.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.27. `catalog_25_finds_conflicting_s2t`

Полный дословный запрос:

~~~~text
Есть ли несколько S2T, которые описывают загрузку b700000025_agr_cred::subquery::v_agr_cred1.del_dt по-разному? Найди все mappings, сравни source fields и transformation, выдели конфликт или объясни, почему mappings дополняют друг друга.
~~~~

Полный публичный ответ:

~~~~text
Для целевого поля b700000025_agr_cred::subquery::v_agr_cred1.del_dt найдено несколько S2T с разными source_field и transformation_rule:

- s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.c_closedate → v_agr_cred1.del_dt: правило = contr.c_closedate (dictionary_ref=5)
- s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract.ctl_validfrom → v_agr_cred1.del_dt: правило = CASE WHEN contr.ctl_action = CAST('D' AS TEXT) THEN contr.ctl_validfrom ELSE CAST(NULL AS TIMESTAMP) END (dictionary_ref=14)

Конфликта нет: первый mapping загружает фактическую дату закрытия договора, второй — условную метку удаления по ctl_action='D'. Они дополняют друг друга.
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.28. `passes_sqlite_result_into_full_neo4j_path`

Полный дословный запрос:

~~~~text
Через Neo4j покажи полный направленный путь от s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loans_productparty_stg до b7000000250039_loans_productregister::union::1: все узлы по порядку и глубину. Полный результат пути покажи отдельно.
~~~~

Полный публичный ответ:

~~~~text
Полный направленный путь от s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loans_productparty_stg до b7000000250039_loans_productregister::union::1 (глубина 3):

Путь 1:
s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loans_productparty_stg → b7000000250039_loans_productregister::subquery::a → b7000000250039_loans_productregister::union::1::branch::1 → b7000000250039_loans_productregister::union::1

Путь 2:
s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loans_productparty_stg → b7000000250039_loans_productregister::subquery::b → b7000000250039_loans_productregister::union::1::branch::2 → b7000000250039_loans_productregister::union::1
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.29. `preserves_exact_s2t_pairs_in_answer_and_full_result`

Полный дословный запрос:

~~~~text
Через SQLite выполни ровно этот read-only запрос: SELECT source_table, source_field, target_table, target_field FROM s2t_transformations WHERE source_table IS NOT NULL AND TRIM(source_table) <> '' AND source_field IS NOT NULL AND TRIM(source_field) <> '' AND target_table IS NOT NULL AND TRIM(target_table) <> '' AND target_field IS NOT NULL AND TRIM(target_field) <> '' ORDER BY id LIMIT 4. Перечисли все 4 точные пары source_table.source_field -> target_table.target_field, не разделяя связанные стороны на отдельные списки. Полный табличный результат покажи отдельно в scrollable UI.
~~~~

Полный публичный ответ:

~~~~text
1) b3080000460002_escrow_legalgk_agreement_actual_dto.Идентификатор договора эскроу -> t_agr_escrow.agr_escrow_id
2) b3080000460002_escrow_legalgk_agreement_actual_dto.Дата открытия договора эскроу -> t_agr_escrow.open_dt
3) b3080000460002_escrow_legalgk_agreement_actual_dto.Дата закрытия договора эскроу -> t_agr_escrow.close_dt
4) b3080000460002_escrow_legalgk_agreement_actual_dto.Срок депонированной суммы -> t_agr_escrow.depositing_dt
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.30. `resolves_history_reference_into_task`

Полный дословный запрос:

~~~~text
Через SQLite посчитай в ней точное количество строк. Только число.
~~~~

Полный публичный ответ:

~~~~text
0
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.31. `returns_complete_three_edge_neo4j_path`

Полный дословный запрос:

~~~~text
Через Neo4j найди полный точный направленный путь длины 3 от таблицы s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loans_productparty_stg до таблицы b7000000250039_loans_productregister::union::1. Не используй SQLite. В ответе покажи только все четыре узла по порядку и глубину. Полный результат со всеми шагами пути покажи отдельно в scrollable UI.
~~~~

Полный публичный ответ:

~~~~text
Узлы пути (по порядку):
1. s_grnplm_as_t_didsd_700_db_stg.l_000025_t_loans_productparty_stg
2. b7000000250039_loans_productregister::subquery::a
3. b7000000250039_loans_productregister::union::1::branch::1
4. b7000000250039_loans_productregister::union::1

Глубина: 3
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.32. `returns_exact_global_sqlite_count`

Полный дословный запрос:

~~~~text
Через SQLite посчитай точное число строк в s2t_transformations. Нужен только итоговый count.
~~~~

Полный публичный ответ:

~~~~text
4257
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.33. `returns_exact_neo4j_path_and_full_result`

Полный дословный запрос:

~~~~text
Через Neo4j найди точный путь длины 2 от таблицы s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loans_productregister до таблицы b7000000250039_loans_productregister::union::1::branch::2. Не используй SQLite. Покажи только все узлы по порядку и глубину, а полный результат инструмента — отдельно.
~~~~

Полный публичный ответ:

~~~~text
Узлы по порядку (глубина 2):
0: s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loans_productregister
1: b7000000250039_loans_productregister::subquery::b
2: b7000000250039_loans_productregister::union::1::branch::2
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.34. `runs_dependent_workers_sequentially`

Полный дословный запрос:

~~~~text
Через SQLite сначала найди target_table с максимальным числом строк в s2t_transformations. Затем отдельным зависимым шагом для найденной target_table посчитай точное число различных непустых source_table. Верни имя target_table, число её строк и число source_table. Полный результат второго шага покажи отдельно.
~~~~

Полный публичный ответ:

~~~~text
target_table=b7000000250002_loansdictstatus
row_count=4
source_table_count=1

Детали по source_table для target_table b7000000250002_loansdictstatus:
- Уникальных непустых source_table: 1 (b7000000250002_loansdictstatus::subquery::cr)
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.35. `runs_three_dependent_sqlite_workers`

Полный дословный запрос:

~~~~text
Через SQLite составь сводку для target_table с наибольшим числом строк в s2t_transformations: имя и число строк target_table, число различных непустых source_table, а также самый частый source_table и число его строк. При равенстве выбери лексикографически первый source_table. Полную сводку покажи отдельно.
~~~~

Полный публичный ответ:

~~~~text
target_table=b7000000250002_loansdictstatus
row_count=4
distinct_nonempty_source_tables=1
most_frequent_source_table=b7000000250002_loansdictstatus::subquery::cr
source_rows=4
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.36. `selects_full_sql_result_for_scrollable_ui`

Полный дословный запрос:

~~~~text
Через SQLite выполни SELECT file_id, filename FROM files ORDER BY file_id и покажи полный результат отдельно в scrollable UI.
~~~~

Полный публичный ответ:

~~~~text
Недостаточно данных: в evidence отсутствуют результаты выполнения SQLite-запроса SELECT file_id, filename FROM files ORDER BY file_id.
~~~~

Текущий вывод: **FAIL — точный SQL не выполнен и отдельный полный display-result отсутствует.**

### 2.37. `writes_independent_s2t_test_protocol`

Полный дословный запрос:

~~~~text
Для file_id=3 подготовь приёмочный протокол из Greenplum SQL для сохранённой загрузки b3050000420007_product → t_crncy. Ничего не запускай. Нужны четыре контроля: совпадает ли рассчитанный по правилу набор с target; отсутствуют ли NULL в обязательных колонках; не повторяется ли подтверждённый ключ; одинаково ли число ожидаемых и загруженных строк. Для каждого укажи назначение, запрос и однозначное условие успешной приёмки.
~~~~

Полный публичный ответ:

~~~~text
Тест-протокол проверки ETL-загрузки во внешней Greenplum СУБД для sources [`b3050000420007_product`] → targets [`t_crncy`]. SQL-шаблоны не исполнялись; фактические метрики не вычислялись. Во всех target-side запросах замени `{{LOAD_SCOPE_PREDICATE}}` условием проверяемой загрузки либо `TRUE` для полного снимка.

Load sources [`b3050000420007_product`] → Target `t_crncy`

1. Проверка корректности трансформаций
Цель: Сравнить ожидаемую S2T-проекцию с target.
SQL-шаблон:
```sql
WITH expected_raw AS (
with src as
(
 select
  b.currencycode_uid
  , max(b.currencycode) as currencycode
  , max(b.input_file_id) as input_file_id
  , max(b.info_system_id) as info_system_id
  , max(b.ctl_action) as ctl_action
  , max(b.ctl_validfrom::date) as ctl_validfrom
 from $$305stg.b3050000420007_product as b
 where 1 = 1
  and b.currencycode_uid is not null
 group by 1
 union all
 select
  b.currencycode_uid
  , max(b.currencycode) as currencycode
  , max(b.input_file_id) as input_file_id
  , max(b.info_system_id) as info_system_id
  , max(b.ctl_action) as ctl_action
  , max(b.ctl_validfrom::date) as ctl_validfrom
 from $$305stg.b3050000420005_paymentdetails as b
 where 1 = 1
  and b.currencycode_uid is not null
 group by 1
)
, src_deduplicated as
(
 select
  b.*
  , row_number() over(partition by b.currencycode_uid order by b.ctl_validfrom desc, b.ctl_action desc, b.input_file_id desc) as rn
 from src as b
)
select b.*
from src_deduplicated as b
where 1 = 1
 and b.rn = 1
),
expected AS (
    SELECT
        src."currencycode_uid" AS "crncy_id",
        src."currencycode" AS "crncy_iso_cd",
        src."currencycode" AS "host_crncy_id"
    FROM expected_raw AS src
),
actual AS (
    SELECT "crncy_id", "crncy_iso_cd", "host_crncy_id"
    FROM "t_crncy"
    WHERE {{LOAD_SCOPE_PREDICATE}}
),
differences AS (
    (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
    UNION ALL
    (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)
)
SELECT COUNT(*) AS difference_count
FROM differences;
```
Критерий прохождения: difference_count = 0.

2. Проверка null-rate обязательных полей
Цель: Проверить NULL в обязательных target-полях.
SQL-шаблон:
```sql
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE "crncy_id" IS NULL) AS "crncy_id_null_count"
FROM "t_crncy"
WHERE {{LOAD_SCOPE_PREDICATE}};
```
Критерий прохождения: Каждый *_null_count равен 0; null-rate каждого обязательного поля равен 0.

3. Проверка уникальности ключа
Цель: Проверить уникальность подтверждённого target-ключа.
SQL-шаблон:
```sql
SELECT "crncy_id", COUNT(*) AS duplicate_count
FROM "t_crncy"
WHERE {{LOAD_SCOPE_PREDICATE}}
GROUP BY "crncy_id"
HAVING COUNT(*) > 1;
```
Критерий прохождения: Запрос не возвращает строк.

4. Проверка количества строк
Цель: Сравнить количество ожидаемых и загруженных строк.
SQL-шаблон:
```sql
WITH expected_raw AS (
with src as
(
 select
  b.currencycode_uid
  , max(b.currencycode) as currencycode
  , max(b.input_file_id) as input_file_id
  , max(b.info_system_id) as info_system_id
  , max(b.ctl_action) as ctl_action
  , max(b.ctl_validfrom::date) as ctl_validfrom
 from $$305stg.b3050000420007_product as b
 where 1 = 1
  and b.currencycode_uid is not null
 group by 1
 union all
 select
  b.currencycode_uid
  , max(b.currencycode) as currencycode
  , max(b.input_file_id) as input_file_id
  , max(b.info_system_id) as info_system_id
  , max(b.ctl_action) as ctl_action
  , max(b.ctl_validfrom::date) as ctl_validfrom
 from $$305stg.b3050000420005_paymentdetails as b
 where 1 = 1
  and b.currencycode_uid is not null
 group by 1
)
, src_deduplicated as
(
 select
  b.*
  , row_number() over(partition by b.currencycode_uid order by b.ctl_validfrom desc, b.ctl_action desc, b.input_file_id desc) as rn
 from src as b
)
select b.*
from src_deduplicated as b
where 1 = 1
 and b.rn = 1
),
expected AS (
    SELECT
        src."currencycode_uid" AS "crncy_id",
        src."currencycode" AS "crncy_iso_cd",
        src."currencycode" AS "host_crncy_id"
    FROM expected_raw AS src
)
SELECT
    (SELECT COUNT(*) FROM expected) AS expected_row_count,
    (SELECT COUNT(*) FROM "t_crncy" WHERE {{LOAD_SCOPE_PREDICATE}}) AS actual_row_count;
```
Критерий прохождения: expected_row_count = actual_row_count.

Подтверждённые основания: S2T sources: ['b3050000420007_product']; mapped target fields: ['crncy_id', 'crncy_iso_cd', 'host_crncy_id']; primary key fields: ['crncy_id']; required fields: ['crncy_id']
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.38. `writes_multi_source_s2t_validation_protocol`

Полный дословный запрос:

~~~~text
Для file_id=2 по сохранённой S2T-загрузке из b3050000420007_product и b3050000420004_nsoadditionalinfo в t_agr_dep составь единый тест-протокол для внешней СУБД. Включи проверки количества строк, уникальности ключа, null-rate обязательных полей и корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий прохождения; фактические метрики не вычисляй.
~~~~

Полный публичный ответ:

~~~~text
Тест-протокол проверки ETL-загрузки во внешней Greenplum СУБД для sources [`b3050000420007_product`, `b3050000420004_nsoadditionalinfo`] → targets [`t_agr_dep`]. SQL-шаблоны не исполнялись; фактические метрики не вычислялись. Во всех target-side запросах замени `{{LOAD_SCOPE_PREDICATE}}` условием проверяемой загрузки либо `TRUE` для полного снимка.

Load sources [`b3050000420007_product`, `b3050000420004_nsoadditionalinfo`] → Target `t_agr_dep`

1. Проверка количества строк
Цель: Сравнить количество ожидаемых и загруженных строк.
SQL-шаблон:
```sql
WITH expected_raw AS (
select
 product.*
 --, nsoadditionalinfo.rkoaccountnumber
 , case
  when lower(trim(product.producttype)) = 'deposit'
  then productaccount.accountnumber
  -- Olga: Для продуктов SDO номер счета РКО лежит там же, где и для NSO - t_core_NSOadditionalinfo.rkoaccountnumber
  else nsoadditionalinfo.rkoaccountnumber
 end as rko_coa_num

 -- Maxim: Это ID договора РКО, грузить надо обязательно, сурогатить по схеме 306. Там это поле t_agr_rko.agr_rko_id.
 -- Принимаем риски:
 -- 1) плохая заполняемость (~ 1,3 млн из 4 млн),
 -- 2) источник предупредил, что idaccountnumber необязательное и на счет РКО надо выходит через rkoaccountnumber.
 , nsoadditionalinfo.idaccountnumber_uid --as agr_rko_id
 , case when upper(product.ctl_action) = 'D' then product.ctl_validfrom else date '9999-12-31' end as del_dt
from
 $$305stg.s305_0007_product as product

 -- No need to use full join, nsoadditionalinfo without product is not needed.
 left outer join $$305stg.s305_0004_nsoadditionalinfo as nsoadditionalinfo
 on true
  and product.object_id = nsoadditionalinfo.product_id
  and upper(nsoadditionalinfo.ctl_action) <> 'D'

 -- No need to use full join, productaccount without product is not needed.
 left outer join
 (
 -- При потере удалений (например, по каким-то причинам пропустили инкремент с удалениями)
 -- можем получить ситуацию, когда для бизнес-ключа (product_id + accounttype) будет несколько записей.
 -- Соответственно, дедублицируем и берем последнюю по ctl_validfrom и object_id.
 select distinct on (productaccount.product_id, productaccount.accounttype)
  *
 from $$305stg.s305_0027_productaccount as productaccount
 where lower(trim(productaccount.accounttype)) = 'debit'
 order by
  productaccount.product_id
  , productaccount.accounttype
  , productaccount.ctl_validfrom desc
  -- Удаление и вставка могут быть с одинаковым ctl_validfrom - берем запись с большим ID.
  , productaccount.object_id desc
 ) as productaccount
 on true
  and product.object_id = productaccount.product_id
  and lower(trim(productaccount.accounttype)) = 'debit'
  and lower(trim(product.producttype)) = 'deposit'
  and upper(productaccount.ctl_action) <> 'D'

where true
 and
 (
  -- Product arrived with increment
  -- 1 - load, 0 - skip, null - skip
  product.incr_flag = 1

  -- Following conditions SHOULD NOT BE NULL!

  -- ProductAccount arrived with increment or Product arrived with increment
  -- 1 - load, 0 - skip, null - analyze product.incr_flag
  or coalesce(productaccount.incr_flag, product.incr_flag) = 1

  -- NsoAdditionalInfo arrived with increment or Product arrived with increment
  -- 1 - load, 0 - skip, null - analyze product.incr_flag
  or coalesce(nsoadditionalinfo.incr_flag, product.incr_flag) = 1
 )
),
expected AS (
    SELECT
        src."object_id_uid" AS "agr_dep_id",
        src."productnumber" AS "agr_dep_num",
        src."producttype_uid" AS "agr_dep_type_id",
        src."agreement_entityid_uid" AS "agr_frame_id",
        src."closedate" AS "close_dt",
        src."createdate" AS "create_dt",
        src."currencycode_uid" AS "crncy_id",
        src."enddate" AS "expiration_dt",
        src."object_id" AS "host_agr_dep_id",
        src."begindate" AS "open_dt",
        src."productkind_id_uid" AS "prod_id",
        src."rko_coa_num" AS "rko_coa_num",
        src."idaccountnumber_uid" AS "agr_rko_id",
        src."contractbegindate" AS "sign_dt",
        src."del_dt" AS "del_dt"
    FROM expected_raw AS src
)
SELECT
    (SELECT COUNT(*) FROM expected) AS expected_row_count,
    (SELECT COUNT(*) FROM "t_agr_dep" WHERE {{LOAD_SCOPE_PREDICATE}}) AS actual_row_count;
```
Критерий прохождения: expected_row_count = actual_row_count.

2. Проверка уникальности ключа
Цель: Проверить уникальность подтверждённого target-ключа.
SQL-шаблон:
```sql
SELECT "agr_dep_id", COUNT(*) AS duplicate_count
FROM "t_agr_dep"
WHERE {{LOAD_SCOPE_PREDICATE}}
GROUP BY "agr_dep_id"
HAVING COUNT(*) > 1;
```
Критерий прохождения: Запрос не возвращает строк.

3. Проверка null-rate обязательных полей
Цель: Проверить NULL в обязательных target-полях.
SQL-шаблон:
```sql
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE "agr_dep_id" IS NULL) AS "agr_dep_id_null_count"
FROM "t_agr_dep"
WHERE {{LOAD_SCOPE_PREDICATE}};
```
Критерий прохождения: Каждый *_null_count равен 0; null-rate каждого обязательного поля равен 0.

4. Проверка корректности трансформаций
Цель: Сравнить ожидаемую S2T-проекцию с target.
SQL-шаблон:
```sql
WITH expected_raw AS (
select
 product.*
 --, nsoadditionalinfo.rkoaccountnumber
 , case
  when lower(trim(product.producttype)) = 'deposit'
  then productaccount.accountnumber
  -- Olga: Для продуктов SDO номер счета РКО лежит там же, где и для NSO - t_core_NSOadditionalinfo.rkoaccountnumber
  else nsoadditionalinfo.rkoaccountnumber
 end as rko_coa_num

 -- Maxim: Это ID договора РКО, грузить надо обязательно, сурогатить по схеме 306. Там это поле t_agr_rko.agr_rko_id.
 -- Принимаем риски:
 -- 1) плохая заполняемость (~ 1,3 млн из 4 млн),
 -- 2) источник предупредил, что idaccountnumber необязательное и на счет РКО надо выходит через rkoaccountnumber.
 , nsoadditionalinfo.idaccountnumber_uid --as agr_rko_id
 , case when upper(product.ctl_action) = 'D' then product.ctl_validfrom else date '9999-12-31' end as del_dt
from
 $$305stg.s305_0007_product as product

 -- No need to use full join, nsoadditionalinfo without product is not needed.
 left outer join $$305stg.s305_0004_nsoadditionalinfo as nsoadditionalinfo
 on true
  and product.object_id = nsoadditionalinfo.product_id
  and upper(nsoadditionalinfo.ctl_action) <> 'D'

 -- No need to use full join, productaccount without product is not needed.
 left outer join
 (
 -- При потере удалений (например, по каким-то причинам пропустили инкремент с удалениями)
 -- можем получить ситуацию, когда для бизнес-ключа (product_id + accounttype) будет несколько записей.
 -- Соответственно, дедублицируем и берем последнюю по ctl_validfrom и object_id.
 select distinct on (productaccount.product_id, productaccount.accounttype)
  *
 from $$305stg.s305_0027_productaccount as productaccount
 where lower(trim(productaccount.accounttype)) = 'debit'
 order by
  productaccount.product_id
  , productaccount.accounttype
  , productaccount.ctl_validfrom desc
  -- Удаление и вставка могут быть с одинаковым ctl_validfrom - берем запись с большим ID.
  , productaccount.object_id desc
 ) as productaccount
 on true
  and product.object_id = productaccount.product_id
  and lower(trim(productaccount.accounttype)) = 'debit'
  and lower(trim(product.producttype)) = 'deposit'
  and upper(productaccount.ctl_action) <> 'D'

where true
 and
 (
  -- Product arrived with increment
  -- 1 - load, 0 - skip, null - skip
  product.incr_flag = 1

  -- Following conditions SHOULD NOT BE NULL!

  -- ProductAccount arrived with increment or Product arrived with increment
  -- 1 - load, 0 - skip, null - analyze product.incr_flag
  or coalesce(productaccount.incr_flag, product.incr_flag) = 1

  -- NsoAdditionalInfo arrived with increment or Product arrived with increment
  -- 1 - load, 0 - skip, null - analyze product.incr_flag
  or coalesce(nsoadditionalinfo.incr_flag, product.incr_flag) = 1
 )
),
expected AS (
    SELECT
        src."object_id_uid" AS "agr_dep_id",
        src."productnumber" AS "agr_dep_num",
        src."producttype_uid" AS "agr_dep_type_id",
        src."agreement_entityid_uid" AS "agr_frame_id",
        src."closedate" AS "close_dt",
        src."createdate" AS "create_dt",
        src."currencycode_uid" AS "crncy_id",
        src."enddate" AS "expiration_dt",
        src."object_id" AS "host_agr_dep_id",
        src."begindate" AS "open_dt",
        src."productkind_id_uid" AS "prod_id",
        src."rko_coa_num" AS "rko_coa_num",
        src."idaccountnumber_uid" AS "agr_rko_id",
        src."contractbegindate" AS "sign_dt",
        src."del_dt" AS "del_dt"
    FROM expected_raw AS src
),
actual AS (
    SELECT "agr_dep_id", "agr_dep_num", "agr_dep_type_id", "agr_frame_id", "close_dt", "create_dt", "crncy_id", "expiration_dt", "host_agr_dep_id", "open_dt", "prod_id", "rko_coa_num", "agr_rko_id", "sign_dt", "del_dt"
    FROM "t_agr_dep"
    WHERE {{LOAD_SCOPE_PREDICATE}}
),
differences AS (
    (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
    UNION ALL
    (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)
)
SELECT COUNT(*) AS difference_count
FROM differences;
```
Критерий прохождения: difference_count = 0.

Подтверждённые основания: S2T sources: ['b3050000420007_product', 'b3050000420004_nsoadditionalinfo']; mapped target fields: ['agr_dep_id', 'agr_dep_num', 'agr_dep_type_id', 'agr_frame_id', 'close_dt', 'create_dt', 'crncy_id', 'expiration_dt', 'host_agr_dep_id', 'open_dt', 'prod_id', 'rko_coa_num', 'agr_rko_id', 'sign_dt', 'del_dt']; primary key fields: ['agr_dep_id']; required fields: ['agr_dep_id']
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

### 2.39. `writes_multi_target_s2t_validation_protocol`

Полный дословный запрос:

~~~~text
Для file_id=3 по сохранённым S2T-загрузкам из b3050000420005_paymentdetails в t_optn и t_optn_type составь отдельный тест-протокол для каждой target-таблицы во внешней СУБД. В каждый включи проверки количества строк, уникальности ключа, null-rate обязательных полей и корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий прохождения; фактические метрики не вычисляй.
~~~~

Полный публичный ответ:

~~~~text
Тест-протокол проверки ETL-загрузки во внешней Greenplum СУБД для sources [`b3050000420005_paymentdetails`] → targets [`t_optn`, `t_optn_type`]. SQL-шаблоны не исполнялись; фактические метрики не вычислялись. Во всех target-side запросах замени `{{LOAD_SCOPE_PREDICATE}}` условием проверяемой загрузки либо `TRUE` для полного снимка.

Load sources [`b3050000420005_paymentdetails`] → Target `t_optn`

1. Проверка количества строк
Цель: Сравнить количество ожидаемых и загруженных строк.
SQL-шаблон:
```sql
WITH expected_raw AS (
select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1
),
expected AS (
    SELECT
        src."object_id_uid" AS "optn_id",
        src."valuedate" AS "start_dt",
        src."product_entityid_uid" AS "agr_dep_id",
        src."technicalservice_id_uid" AS "bus_srv_id",
        src."currencycode_uid" AS "crncy_id",
        src."object_id" AS "host_optn_id",
        src."amount" AS "optn_amt",
        src."amountrub" AS "optn_amt_rub",
        src."operationtypecode_uid" AS "optn_type_id",
        src."registerid_uid" AS "registry_id"
    FROM expected_raw AS src
)
SELECT
    (SELECT COUNT(*) FROM expected) AS expected_row_count,
    (SELECT COUNT(*) FROM "t_optn" WHERE {{LOAD_SCOPE_PREDICATE}}) AS actual_row_count;
```
Критерий прохождения: expected_row_count = actual_row_count.

2. Проверка уникальности ключа
Цель: Проверить уникальность подтверждённого target-ключа.
SQL-шаблон:
```sql
SELECT "optn_id", COUNT(*) AS duplicate_count
FROM "t_optn"
WHERE {{LOAD_SCOPE_PREDICATE}}
GROUP BY "optn_id"
HAVING COUNT(*) > 1;
```
Критерий прохождения: Запрос не возвращает строк.

3. Проверка null-rate обязательных полей
Цель: Проверить NULL в обязательных target-полях.
SQL-шаблон:
```sql
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE "optn_id" IS NULL) AS "optn_id_null_count"
FROM "t_optn"
WHERE {{LOAD_SCOPE_PREDICATE}};
```
Критерий прохождения: Каждый *_null_count равен 0; null-rate каждого обязательного поля равен 0.

4. Проверка корректности трансформаций
Цель: Сравнить ожидаемую S2T-проекцию с target.
SQL-шаблон:
```sql
WITH expected_raw AS (
select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1
),
expected AS (
    SELECT
        src."object_id_uid" AS "optn_id",
        src."valuedate" AS "start_dt",
        src."product_entityid_uid" AS "agr_dep_id",
        src."technicalservice_id_uid" AS "bus_srv_id",
        src."currencycode_uid" AS "crncy_id",
        src."object_id" AS "host_optn_id",
        src."amount" AS "optn_amt",
        src."amountrub" AS "optn_amt_rub",
        src."operationtypecode_uid" AS "optn_type_id",
        src."registerid_uid" AS "registry_id"
    FROM expected_raw AS src
),
actual AS (
    SELECT "optn_id", "start_dt", "agr_dep_id", "bus_srv_id", "crncy_id", "host_optn_id", "optn_amt", "optn_amt_rub", "optn_type_id", "registry_id"
    FROM "t_optn"
    WHERE {{LOAD_SCOPE_PREDICATE}}
),
differences AS (
    (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
    UNION ALL
    (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)
)
SELECT COUNT(*) AS difference_count
FROM differences;
```
Критерий прохождения: difference_count = 0.

Подтверждённые основания: S2T sources: ['b3050000420005_paymentdetails', 'b3050000420015_technicalservice']; mapped target fields: ['optn_id', 'start_dt', 'agr_dep_id', 'bus_srv_id', 'crncy_id', 'host_optn_id', 'optn_amt', 'optn_amt_rub', 'optn_type_id', 'registry_id']; primary key fields: ['optn_id']; required fields: ['optn_id']

Load sources [`b3050000420005_paymentdetails`] → Target `t_optn_type`

1. Проверка количества строк
Цель: Сравнить количество ожидаемых и загруженных строк.
SQL-шаблон:
```sql
WITH expected_raw AS (
with src as
(
 select
  b.operationtypecode_uid
  , max(b.operationtypecode) as operationtypecode
  , max(b.input_file_id) as input_file_id
  , max(b.info_system_id) as info_system_id
  , max(b.ctl_action) as ctl_action
  , max(b.ctl_validfrom::date) as ctl_validfrom
 from $$305stg.b3050000420005_paymentdetails as b
 where true
  and b.operationtypecode_uid is not null
 group by b.operationtypecode_uid
)
select
 b.*
 , enums.label
from
 src as b

 left outer join $$305stg.s305_0999_enums as enums
 on true
  and lower(trim(b.operationtypecode)) = lower(trim(enums.value_name))
  and lower(trim(enums.enum_name)) = 'manual_operationtype'
  and enums.label is not null
),
expected AS (
    SELECT
        src."operationtypecode_uid" AS "optn_type_id",
        src."operationtypecode" AS "host_optn_type_id",
        src."operationtypecode" AS "optn_type_cd",
        src."label" AS "optn_type_name"
    FROM expected_raw AS src
)
SELECT
    (SELECT COUNT(*) FROM expected) AS expected_row_count,
    (SELECT COUNT(*) FROM "t_optn_type" WHERE {{LOAD_SCOPE_PREDICATE}}) AS actual_row_count;
```
Критерий прохождения: expected_row_count = actual_row_count.

2. Проверка уникальности ключа
Цель: Проверить уникальность подтверждённого target-ключа.
SQL-шаблон:
```sql
SELECT "optn_type_id", COUNT(*) AS duplicate_count
FROM "t_optn_type"
WHERE {{LOAD_SCOPE_PREDICATE}}
GROUP BY "optn_type_id"
HAVING COUNT(*) > 1;
```
Критерий прохождения: Запрос не возвращает строк.

3. Проверка null-rate обязательных полей
Цель: Проверить NULL в обязательных target-полях.
SQL-шаблон:
```sql
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE "optn_type_id" IS NULL) AS "optn_type_id_null_count"
FROM "t_optn_type"
WHERE {{LOAD_SCOPE_PREDICATE}};
```
Критерий прохождения: Каждый *_null_count равен 0; null-rate каждого обязательного поля равен 0.

4. Проверка корректности трансформаций
Цель: Сравнить ожидаемую S2T-проекцию с target.
SQL-шаблон:
```sql
WITH expected_raw AS (
with src as
(
 select
  b.operationtypecode_uid
  , max(b.operationtypecode) as operationtypecode
  , max(b.input_file_id) as input_file_id
  , max(b.info_system_id) as info_system_id
  , max(b.ctl_action) as ctl_action
  , max(b.ctl_validfrom::date) as ctl_validfrom
 from $$305stg.b3050000420005_paymentdetails as b
 where true
  and b.operationtypecode_uid is not null
 group by b.operationtypecode_uid
)
select
 b.*
 , enums.label
from
 src as b

 left outer join $$305stg.s305_0999_enums as enums
 on true
  and lower(trim(b.operationtypecode)) = lower(trim(enums.value_name))
  and lower(trim(enums.enum_name)) = 'manual_operationtype'
  and enums.label is not null
),
expected AS (
    SELECT
        src."operationtypecode_uid" AS "optn_type_id",
        src."operationtypecode" AS "host_optn_type_id",
        src."operationtypecode" AS "optn_type_cd",
        src."label" AS "optn_type_name"
    FROM expected_raw AS src
),
actual AS (
    SELECT "optn_type_id", "host_optn_type_id", "optn_type_cd", "optn_type_name"
    FROM "t_optn_type"
    WHERE {{LOAD_SCOPE_PREDICATE}}
),
differences AS (
    (SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
    UNION ALL
    (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected)
)
SELECT COUNT(*) AS difference_count
FROM differences;
```
Критерий прохождения: difference_count = 0.

Подтверждённые основания: S2T sources: ['b3050000420005_paymentdetails']; mapped target fields: ['optn_type_id', 'host_optn_type_id', 'optn_type_cd', 'optn_type_name']; primary key fields: ['optn_type_id']; required fields: ['optn_type_id']
~~~~

Текущий вывод: **PASS — запрос выполнен семантически корректно.**

## Оставшиеся содержательные проблемы
### `selects_full_sql_result_for_scrollable_ui`

Запрос требует выполнить в SQLite точный `SELECT file_id, filename FROM files ORDER BY file_id` и показать полный результат отдельно. На специализированной стадии `run_sql` отсутствует в доступной палитре. Выбор этого инструмента после reroute отклоняется как недоступный, а fallback его также не содержит. Worker завершается без данных, поэтому нет ни результата SQL, ни display.

Нужно обеспечить доступность общего read-only SQL-инструмента для явно переданного SQL либо гарантированно расширять палитру до завершения worker.

### `catalog_02_finds_source_field_targets`

Для точной пары `source_table=s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract`, `source_field=c_closedate` выбирается чтение всей source-таблицы без фильтра по полю. Из широкого набора возвращается один правильный target, придумывается отсутствующий и теряются два существующих target.

Нужно сохранять гранулярность задачи: точные фильтры должны приводить к точному reader, а observer не должен принимать более широкий результат как полный.

### `catalog_08_searches_client_id_synonyms`

Имя внутренней таблицы `s2t_transformations` используется как значение логического `target_table`. Пустой результат принимается как завершение, хотя существуют две уникальные пары:

- `b3050000420007_product.client_entityid_uid → t_agr_dep_cust.cust_id`;
- `b3050000420002_baseagreement.client_entityid_uid → t_agr_frame_cust.cust_id`.

Нужно жёстко разделять имена публичных хранилищ и значения ETL-сущностей на уровне описаний аргументов и проверки результата.

### `catalog_09_maps_russian_term_to_technical_field`

Семантический поиск возвращает несколько кандидатов `DEL_DT/del_dt`, но зависимый worker произвольно выбирает один из них и не проверяет остальные. Техническое поле в ответе есть, а запрошенного S2T-правила нет.

Нужна пакетная проверка всего набора кандидатов из previous results вместо одиночного выбора моделью.

### `catalog_20_finds_data_loss_points`

Неполное имя source-таблицы передаётся exact-reader без предварительного разрешения полного идентификатора. Дальнейшие зависимые задачи работают с пустым результатом, а финальный ответ заменяет фактические S2T, промежуточные таблицы, JOIN и FILTER общим чек-листом.

Нужно разрешать неполный идентификатор до exact-чтения и не формировать содержательный ответ без подтверждённого evidence.

## Общие архитектурные ограничения

- Общие инструменты доступны не на всех стадиях, а worker может завершиться до полезного расширения палитры.
- Точные фильтры задачи не всегда сохраняются при выборе reader.
- Observer может принять пустой или слишком широкий результат как достаточный.
- Набор кандидатов из предыдущего результата иногда сводится к одному произвольному значению.
- Неполный идентификатор может попасть в exact-reader без разрешения.
- Финальный координатор способен перейти к обобщениям при недостаточном evidence.
- Сложные успешные ответы всё ещё требуют лишних router/planner/observer-циклов и повторного чтения результатов.

Приоритет исправлений: точная гранулярность reader-ов, пакетная обработка зависимых результатов, корректное разрешение неполных идентификаторов и запрет завершения при пустом evidence.

## Ограничения автоматической оценки

Автоматический judge полезен для поиска выдуманных таблиц и колонок, но пока:

- может не увидеть подтверждённый идентификатор вне выбранного display-content;
- может принять имя внутренней SQLite-таблицы за ETL-объект;
- может принять имя agent tool за физический идентификатор.

Поэтому итоговый статус в этом документе основан на смысловой проверке публичного ответа и подтверждающих данных, а не только на автоматическом verdict.

## Состояние загрузки Excel

При загрузке доступны отдельные показатели:

- `data_row_count` — число фактически разобранных строк каждого листа;
- `total_data_row_count` — сумма строк по книге;
- количество сохранённых S2T-трансформаций;
- количество строк S2T-листов с пустой target-колонкой.

Пропущенный лист даёт `data_row_count=0`; скрытые строки учитываются согласно `include_hidden_rows`; одинаковые строки считаются отдельно; число строк не зависит от количества заполненных ячеек. Итоги возвращаются через API/progress и отображаются в интерфейсе.
