# Отчёт по 21 live-прогону основных S2T-сценариев

Дата: 3 сентября 2026 года.

- Модель агента: `GigaChat-3-Ultra`.
- Модель LLM-as-judge: `GigaChat-2-Pro`.
- Режим: `multiagent`, реальные последовательные HTTP-запросы к `/chat`.
- Код: коммит `a2862bd` (`feat: add typed S2T analysis pipelines`).
- В итог включены ровно 21 пригодная попытка: по три на каждый из семи сценариев.
- Попытки с инфраструктурным загрязнением не засчитывались и были заменены.

## Итог

- Полный semantic pass по judge: **18/21**.
- Стабильно полностью проходят: **6/7 сценариев**, каждый по 3/3.
- Сценарий обязательных полей: факты верны 3/3, но ответ 3/3 не содержит явный пустой список `fields_without_mapping=[]`; поэтому judge справедливо считает выполнение требования неполным.
- После исправления display сценарии анализа риска дубликатов и потери строк проходят 3/3: judge видит точные S2T-строки и полное transformation rule.
- Суммарно: 389 658 токенов, 199 LLM-вызовов, 361.631 секунды agent time.

| № | Сценарий | Полные проходы | Ручная оценка |
|---:|---|---:|---|
| 1 | Совместимость nullable-ограничений | 3/3 | Полностью корректно |
| 2 | Совместимость типов | 3/3 | Полностью корректно |
| 3 | Риск дубликатов | 3/3 | Полностью корректно |
| 4 | Обязательные поля без S2T-маппинга | 0/3 | Числа верны; пустой список не напечатан |
| 5 | Риск потери строк | 3/3 | Полностью корректно |
| 6 | Объяснение трансформации | 3/3 | Полностью корректно |
| 7 | Тест-протокол ETL-загрузки | 3/3 | Полностью корректно |

## Состав трёх циклов

- Цикл 1: семь результатов из `run1`.
- Цикл 2: сценарии 2–7 из `run2`; сценарий 1 заменён чистым `retries #1`.
- Цикл 3: сценарии 1 и 3–7 из `run3`; сценарий 2 заменён чистым `retries #2`.

## Исключённые попытки

1. `run2 #1`: HTTP 200 и содержательно правильный ответ, но один внутренний worker-planner вызов завершился ошибкой и оставил нулевую token-метрику. Попытка консервативно исключена.
2. `run3 #2`: HTTP 500 — `ReadTimeout` на downstream-плане; публичного содержательного ответа не было. Попытка исключена.

## Дословные запросы и ответы

### Цикл 1, сценарий 1: Совместимость nullable-ограничений

Источник: `run1 #1`.
HTTP: `200`. Время агента: `63.712` с. LLM-вызовы: `36`. Токены: `87016`.
Display-results: `get_source_target_column_pair, list_target_column_catalog`.

Дословный запрос:

~~~~text
Для file_id=3 оцени совместимость nullable-ограничений b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_not_null=<0|1>, target_not_null=<0|1> и вывод.
~~~~

Полный публичный ответ модели:

~~~~text
source_not_null=0, target_not_null=1, вывод: несовместимо — целевой столбец NOT NULL, а исходный допускает NULL
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Ответ полностью соответствует запросу: указаны source_not_null=0, target_not_null=1 и корректный вывод о несовместимости. Display_results содержат необходимые сведения о nullable-ограничениях для обеих сторон сопоставления.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 1, сценарий 2: Совместимость типов

Источник: `run1 #2`.
HTTP: `200`. Время агента: `14.845` с. LLM-вызовы: `9`. Токены: `14713`.
Display-results: `get_source_target_column_pair`.

Дословный запрос:

~~~~text
Для file_id=3 оцени совместимость типов b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_data_type=<тип>, target_data_type=<тип> и вывод.
~~~~

Полный публичный ответ модели:

~~~~text
source_data_type=uuid, target_data_type=uuid, вывод=Совместимы
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer и display_results точно соответствуют запросу: указаны source_data_type и target_data_type как uuid, вывод о совместимости корректен, display содержит запрошенные source и target колонки с file_id=3 и их типы.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 1, сценарий 3: Риск дубликатов

Источник: `run1 #3`.
HTTP: `200`. Время агента: `3.546` с. LLM-вызовы: `3`. Токены: `2382`.
Display-results: `read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Оцени риск появления дубликатов при сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ модели:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск дубликатов: Найден потенциальный риск размножения строк: кардинальность одного или нескольких JOIN неизвестна.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice; cardinality=unknown
Ограничения: Фактические дубли без физических данных неизвестны.
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Query выполнен полностью: указан анализ S2T-пути, явно оценен риск дубликатов с обоснованием (LEFT OUTER JOIN с unknown cardinality), перечислены source и target таблицы, приведён transformation_rule. Дополнительные физические данные не запрашивались, что соответствует условиям.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 1, сценарий 4: Обязательные поля без S2T-маппинга

Источник: `run1 #4`.
HTTP: `200`. Время агента: `23.473` с. LLM-вызовы: `14`. Токены: `41799`.
Display-results: `list_target_column_catalog, read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Для file_id=3 найди обязательные поля t_optn без сохранённого S2T-маппинга. Верни mandatory_fields_count=<число>, mandatory_fields_without_mapping_count=<число> и имена полей без маппинга.
~~~~

Полный публичный ответ модели:

~~~~text
mandatory_fields_count=1, mandatory_fields_without_mapping_count=0
~~~~

LLM-as-judge: **failed**.

Дословное заключение judge:

~~~~text
Отсутствует перечисление имён полей без S2T-маппинга, хотя это явно запрошено в query.
~~~~

Ручной вывод: Частичное выполнение: оба числа корректны, но явно запрошенный список имён не выведен как пустой список.

### Цикл 1, сценарий 5: Риск потери строк

Источник: `run1 #5`.
HTTP: `200`. Время агента: `3.599` с. LLM-вызовы: `3`. Токены: `2377`.
Display-results: `read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Оцени риск потери строк в сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ модели:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск потери строк: Структурный риск удаления исходных строк не найден: нет WHERE/HAVING и непредохраняющих JOIN.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice сохраняет левые строки
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Query выполнен полностью: риск потери строк в S2T-трансформации оценен, указаны основания, приведён полный S2T-каталог по целевой таблице.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 1, сценарий 6: Объяснение трансформации

Источник: `run1 #6`.
HTTP: `200`. Время агента: `19.501` с. LLM-вызовы: `9`. Токены: `22627`.
Display-results: `read_s2t_source_to_target`.

Дословный запрос:

~~~~text
Объясни сохранённую S2T-трансформацию b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id.
~~~~

Полный публичный ответ модели:

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

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer детально объясняет запрошенную S2T-трансформацию с описанием SQL-правила, роли полей и логики соединения. Display_results содержит таблицу соответствий с явным указанием source_table → target_table и source_field → target_field, включая запрошенную пару object_id_uid → optn_id. Все части запроса выполнены корректно.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 1, сценарий 7: Тест-протокол ETL-загрузки

Источник: `run1 #7`.
HTTP: `200`. Время агента: `5.082` с. LLM-вызовы: `3`. Токены: `2699`.
Display-results: `read_s2t_source_to_target, list_target_column_catalog`.

Дословный запрос:

~~~~text
Для file_id=3 по сохранённой S2T-спецификации b3050000420005_paymentdetails → t_optn составь тест-протокол для проверки ETL-загрузки во внешней СУБД. Включи проверки количества строк, уникальности ключа, null-rate обязательных полей и корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий прохождения. Используй подтверждённые таблицы, колонки и правила; фактические метрики не вычисляй.
~~~~

Полный публичный ответ модели:

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

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Ответ содержит полный тест-протокол с целями, SQL-шаблонами и критериями прохождения для всех запрошенных проверок: количество строк, уникальность ключа, null-rate обязательных полей и корректность трансформаций. Указаны подтверждённые таблицы, поля и правила. Display_results содержат необходимую S2T-спецификацию и целевые колонки, что подтверждает соответствие запросу.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 2, сценарий 1: Совместимость nullable-ограничений

Источник: `retries #1`; замещающая чистая попытка.
HTTP: `200`. Время агента: `38.608` с. LLM-вызовы: `19`. Токены: `31524`.
Display-results: `get_source_target_column_pair`.

Дословный запрос:

~~~~text
Для file_id=3 оцени совместимость nullable-ограничений b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_not_null=<0|1>, target_not_null=<0|1> и вывод.
~~~~

Полный публичный ответ модели:

~~~~text
source_not_null=0, target_not_null=1. Вывод: несовместимо — целевой столбец NOT NULL, а исходный допускает NULL.
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer и display_results корректно отражают nullable-ограничения: source_not_null=0, target_not_null=1, вывод о несовместимости сделан верно. Display содержит запрошенную source→target-пару с нужными атрибутами.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 2, сценарий 2: Совместимость типов

Источник: `run2 #2`.
HTTP: `200`. Время агента: `14.277` с. LLM-вызовы: `9`. Токены: `11758`.
Display-results: `get_source_target_column_pair`.

Дословный запрос:

~~~~text
Для file_id=3 оцени совместимость типов b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_data_type=<тип>, target_data_type=<тип> и вывод.
~~~~

Полный публичный ответ модели:

~~~~text
source_data_type=uuid, target_data_type=uuid, вывод=Совместимы
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer и display_results точно соответствуют запросу: указаны source_data_type и target_data_type как uuid, вывод о совместимости сделан корректно. Display содержит запрошенную пару столбцов с file_id=3 и их типы данных.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 2, сценарий 3: Риск дубликатов

Источник: `run2 #3`.
HTTP: `200`. Время агента: `4.461` с. LLM-вызовы: `3`. Токены: `2382`.
Display-results: `read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Оцени риск появления дубликатов при сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ модели:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск дубликатов: Найден потенциальный риск размножения строк: кардинальность одного или нескольких JOIN неизвестна.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice; cardinality=unknown
Ограничения: Фактические дубли без физических данных неизвестны.
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Query выполнен полностью: анализ S2T-трансформации b3050000420005_paymentdetails → t_optn проведён, риск дубликатов оценён корректно с указанием причины (LEFT OUTER JOIN с unknown cardinality), ограничения сформулированы. Display содержит релевантные детали трансформации.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 2, сценарий 4: Обязательные поля без S2T-маппинга

Источник: `run2 #4`.
HTTP: `200`. Время агента: `24.455` с. LLM-вызовы: `13`. Токены: `33599`.
Display-results: `list_target_column_catalog, read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Для file_id=3 найди обязательные поля t_optn без сохранённого S2T-маппинга. Верни mandatory_fields_count=<число>, mandatory_fields_without_mapping_count=<число> и имена полей без маппинга.
~~~~

Полный публичный ответ модели:

~~~~text
mandatory_fields_count=1, mandatory_fields_without_mapping_count=0
~~~~

LLM-as-judge: **failed**.

Дословное заключение judge:

~~~~text
Отсутствует перечисление имён полей без S2T-маппинга, хотя это прямо запрошено в query.
~~~~

Ручной вывод: Частичное выполнение: оба числа корректны, но явно запрошенный список имён не выведен как пустой список.

### Цикл 2, сценарий 5: Риск потери строк

Источник: `run2 #5`.
HTTP: `200`. Время агента: `3.768` с. LLM-вызовы: `3`. Токены: `821`.
Display-results: `read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Оцени риск потери строк в сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ модели:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск потери строк: Структурный риск удаления исходных строк не найден: нет WHERE/HAVING и непредохраняющих JOIN.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice сохраняет левые строки
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Query требует оценки риска потери строк в S2T-трансформации b3050000420005_paymentdetails → t_optn. Ответ содержит структурный анализ трансформации, явно указывает на отсутствие риска потери строк с обоснованием через тип JOIN и условия. Display_results предоставляет детализацию S2T-преобразований, включая правило трансформации, что подтверждает полноту ответа.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 2, сценарий 6: Объяснение трансформации

Источник: `run2 #6`.
HTTP: `200`. Время агента: `19.84` с. LLM-вызовы: `9`. Токены: `20273`.
Display-results: `read_s2t_source_to_target`.

Дословный запрос:

~~~~text
Объясни сохранённую S2T-трансформацию b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id.
~~~~

Полный публичный ответ модели:

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

where 1 = 1;

Пояснения по ключевым элементам правила:
- Источник: слой B, таблица b3050000420005_paymentdetails; поле object_id_uid.
- Цель: слой T, таблица t_optn; поле optn_id.
- В SELECT включено всё из источника (b.*), что обеспечивает прямое маппинг-поле source_field→target_field без преобразований для данной пары.
- LEFT JOIN с s305_0015_technicalservice не фильтрует строки источника и используется только для обогащения данными технического сервиса; он не влияет на заполнение поля optn_id в целевой таблице.
- Фильтр where 1=1 — нейтральный предикат, фактической фильтрации нет.
- Дополнительное условие UPPER(technicalservice.ctl_action) <> 'D' относится к правой таблице соединения и исключает помеченные на удаление записи технического сервиса при обогащении.
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer объясняет сохранённую S2T-трансформацию с указанием SQL-правила и подробными пояснениями по ключевым элементам. Display_results содержит таблицу с mapping'ами source→target, включая запрошенную пару object_id_uid → optn_id, что полностью удовлетворяет запросу.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 2, сценарий 7: Тест-протокол ETL-загрузки

Источник: `run2 #7`.
HTTP: `200`. Время агента: `4.866` с. LLM-вызовы: `3`. Токены: `2699`.
Display-results: `read_s2t_source_to_target, list_target_column_catalog`.

Дословный запрос:

~~~~text
Для file_id=3 по сохранённой S2T-спецификации b3050000420005_paymentdetails → t_optn составь тест-протокол для проверки ETL-загрузки во внешней СУБД. Включи проверки количества строк, уникальности ключа, null-rate обязательных полей и корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий прохождения. Используй подтверждённые таблицы, колонки и правила; фактические метрики не вычисляй.
~~~~

Полный публичный ответ модели:

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

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Ответ содержит полный тест-протокол с целями, SQL-шаблонами и критериями прохождения для всех запрошенных проверок: количество строк, уникальность ключа, null-rate обязательных полей и корректность трансформаций. Указаны подтверждённые таблицы, поля и правила. Display_results содержат S2T-спецификацию и целевые колонки, что подтверждает соответствие запросу. Все требования query выполнены.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 3, сценарий 1: Совместимость nullable-ограничений

Источник: `run3 #1`.
HTTP: `200`. Время агента: `24.205` с. LLM-вызовы: `13`. Токены: `23704`.
Display-results: `get_source_target_column_pair, list_target_column_catalog`.

Дословный запрос:

~~~~text
Для file_id=3 оцени совместимость nullable-ограничений b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_not_null=<0|1>, target_not_null=<0|1> и вывод.
~~~~

Полный публичный ответ модели:

~~~~text
source_not_null=0, target_not_null=1, вывод: несовместимо — источник допускает NULL, а целевое поле обязательно
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Запрос на оценку совместимости nullable-ограничений выполнен корректно: указаны source_not_null, target_not_null и обоснованный вывод. Display_results содержат необходимые сведения о nullable для обеих колонок.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 3, сценарий 2: Совместимость типов

Источник: `retries #2`; замещающая чистая попытка.
HTTP: `200`. Время агента: `25.363` с. LLM-вызовы: `13`. Токены: `22515`.
Display-results: `get_source_target_column_pair, list_target_column_catalog`.

Дословный запрос:

~~~~text
Для file_id=3 оцени совместимость типов b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id. Верни source_data_type=<тип>, target_data_type=<тип> и вывод.
~~~~

Полный публичный ответ модели:

~~~~text
source_data_type=uuid
target_data_type=uuid
Вывод: типы совместимы (оба uuid).
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer и display_results корректно выполняют запрос: указаны типы source и target (оба uuid), приведён вывод о совместимости, а display_results содержат необходимые сведения о column_role, file_id, table_name и column_name для обеих сторон сопоставления.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 3, сценарий 3: Риск дубликатов

Источник: `run3 #3`.
HTTP: `200`. Время агента: `3.946` с. LLM-вызовы: `3`. Токены: `2382`.
Display-results: `read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Оцени риск появления дубликатов при сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ модели:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск дубликатов: Найден потенциальный риск размножения строк: кардинальность одного или нескольких JOIN неизвестна.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice; cardinality=unknown
Ограничения: Фактические дубли без физических данных неизвестны.
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Query выполнен полностью: указан анализ S2T-пути, оценен риск дубликатов с обоснованием (JOIN с unknown cardinality), приведены детали трансформации и задействованные таблицы. Дополнительные физические данные не запрашивались, что соответствует условиям.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 3, сценарий 4: Обязательные поля без S2T-маппинга

Источник: `run3 #4`.
HTTP: `200`. Время агента: `22.373` с. LLM-вызовы: `13`. Токены: `32867`.
Display-results: `list_target_column_catalog, read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Для file_id=3 найди обязательные поля t_optn без сохранённого S2T-маппинга. Верни mandatory_fields_count=<число>, mandatory_fields_without_mapping_count=<число> и имена полей без маппинга.
~~~~

Полный публичный ответ модели:

~~~~text
mandatory_fields_count=1, mandatory_fields_without_mapping_count=0
~~~~

LLM-as-judge: **failed**.

Дословное заключение judge:

~~~~text
Отсутствует перечисление имён полей без S2T-маппинга, хотя это прямо запрошено в query.
~~~~

Ручной вывод: Частичное выполнение: оба числа корректны, но явно запрошенный список имён не выведен как пустой список.

### Цикл 3, сценарий 5: Риск потери строк

Источник: `run3 #5`.
HTTP: `200`. Время агента: `3.895` с. LLM-вызовы: `3`. Токены: `821`.
Display-results: `read_s2t_by_target_table`.

Дословный запрос:

~~~~text
Оцени риск потери строк в сохранённой S2T-трансформации b3050000420005_paymentdetails → t_optn.
~~~~

Полный публичный ответ модели:

~~~~text
S2T-анализ sources [`b3050000420005_paymentdetails`] → targets [`t_optn`]. Физические данные логических ETL-таблиц не запрашивались.

Target `t_optn`

Риск потери строк: Структурный риск удаления исходных строк не найден: нет WHERE/HAVING и непредохраняющих JOIN.
Основания: LEFT OUTER JOIN "$$305stg".s305_0015_technicalservice AS technicalservice сохраняет левые строки
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Query требует оценки риска потери строк в S2T-трансформации b3050000420005_paymentdetails → t_optn. Ответ содержит структурный анализ трансформации, явно указывает на отсутствие риска потери строк с обоснованием через тип JOIN и условия. Display_results предоставляет детализацию S2T-преобразований, включая правило трансформации, что подтверждает полноту ответа.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 3, сценарий 6: Объяснение трансформации

Источник: `run3 #6`.
HTTP: `200`. Время агента: `31.911` с. LLM-вызовы: `15`. Токены: `28001`.
Display-results: `read_s2t_source_to_target`.

Дословный запрос:

~~~~text
Объясни сохранённую S2T-трансформацию b3050000420005_paymentdetails.object_id_uid → t_optn.optn_id.
~~~~

Полный публичный ответ модели:

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

where 1 = 1;

Пояснение по ключевым элементам правила:
- Источник: слой B, таблица $$305stg.b3050000420005_paymentdetails (алиас b).
- Левое присоединение к справочной таблице s305_0015_technicalservice для обогащения; фильтрация исключает записи с ctl_action='D' (регистр игнорируется через UPPER). Условие ON TRUE делает фильтр и связь независимыми.
- WHERE 1=1 не ограничивает выборку.
- В результирующем наборе присутствуют все поля источника (b.*); целевое поле optn_id в t_optn заполняется из source-поля object_id_uid без выражений преобразования — прямой перенос значения на уровне строкового маппинга S2T.
~~~~

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Answer корректно объясняет S2T-трансформацию с указанием SQL-правила и пояснением ключевых элементов. Display_results предоставляет полный список всех S2T-преобразований между указанными таблицами, включая запрошенное поле object_id_uid → optn_id. Оба компонента удовлетворяют запросу пользователя.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

### Цикл 3, сценарий 7: Тест-протокол ETL-загрузки

Источник: `run3 #7`.
HTTP: `200`. Время агента: `5.905` с. LLM-вызовы: `3`. Токены: `2699`.
Display-results: `read_s2t_source_to_target, list_target_column_catalog`.

Дословный запрос:

~~~~text
Для file_id=3 по сохранённой S2T-спецификации b3050000420005_paymentdetails → t_optn составь тест-протокол для проверки ETL-загрузки во внешней СУБД. Включи проверки количества строк, уникальности ключа, null-rate обязательных полей и корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и критерий прохождения. Используй подтверждённые таблицы, колонки и правила; фактические метрики не вычисляй.
~~~~

Полный публичный ответ модели:

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

LLM-as-judge: **passed**.

Дословное заключение judge:

~~~~text
Ответ содержит полный тест-протокол с целями, SQL-шаблонами и критериями прохождения для всех запрошенных проверок: количество строк, уникальность ключа, null-rate обязательных полей и корректность трансформаций. Указаны подтверждённые таблицы, поля и правила. Display_results содержат S2T-спецификацию и целевые колонки, что подтверждает соответствие запросу. Все требования query выполнены.
~~~~

Ручной вывод: Полное выполнение: факты и вывод соответствуют запросу.

## Заключение

Исправление передачи display-evidence решило исходную наблюдаемую проблему специализированного `s2t_analysis`: полные reader-results теперь доступны judge и пользователю. Содержательные выводы о рисках стали устойчивыми 3/3. Оставшийся дефект локален и не связан с получением данных: финальный renderer ответа по пустому множеству не печатает явное представление списка.

Unit-проверка перед live-прогонами: `94 passed`.
