# Демонстрация ETL-агента

## 1. Непосредственный upstream колонки

### Запрос

> Покажи непосредственный upstream для колонки `t_bus_srv.bus_srv_id`. Нужны
> исходные таблицы, исходные колонки и слои переходов.

### Результат

````text
Для колонки `t_bus_srv.bus_srv_id` найден непосредственный upstream (4 источника):

```table
[["Исходная таблица","source_layer","Колонка-источник","ID трансформации","Целевая таблица","Цель. слой","Cтeль. колонка"],["b3050000420001_agreementservice","B","object_id_uid",585,"t_bus_srv","T","bus_srv_id"],["b3050000420010_productservice","B","object_id_uid",679,"t_bus_srv","T","bus_srv_id"],["b3050000420015_technicalservice","B","object_id_uid",696,"t_bus_srv","T","bus_srv_id"],["b3050000420029_planservice","B","object_id_uid",785,"t_bus_srv","T","bus_srv_id"]]
```

Все переходы — прямые маппинги без дедупликации (`wildcard_passthrough = false`).
````

## 2. Топ целевых ETL-таблиц

### Запрос

> Назови пять целевых ETL-таблиц с наибольшим количеством уникальных
> таблиц-источников. Верни название целевой таблицы и количество источников.

### Результат

````text
```table
["target_table","count_sources"],["B7000000250071_I_AGR_COND_VAL",10],["t_rate_rule_param",5],["t_bus_srv",5],["b7000000250004_loansagreement",5],["b7000000250039_loans_productregister::subquery::b_2",5]
```
````

## 3. Колонковый lineage Greenplum SQL

### Запрос

> Разбери колонковое происхождение данных в этом Greenplum SQL, но не выполняй
> его:
>
> ```sql
> WITH src AS (
>   SELECT o.id, UPPER(o.name) AS normalized_name
>   FROM raw.orders o -- исходные заказы
> )
> SELECT id, normalized_name, 1::BIGINT AS version
> FROM src
> ```

### Результат

````text
Вот разбор колонкового происхождения данных для вашего Greenplum SQL-запроса:

```table
[["target_column","source_table","source_column"],["id","raw.orders","id"],["normalized_name","raw.orders","name"],["version",null,null]]
```

**Объяснение:**
*   **`id`**: берётся напрямую из таблицы `raw.orders`, колонка `id`.
*   **`normalized_name`**: формируется на основе колонки `name` из той же таблицы `raw.orders` с применением функции преобразования в верхний регистр (`UPPER`).
*   **`version`**: является константой (литералом) и не зависит от данных исходных таблиц.
````

## 4. Таблицы, выступающие в обеих ролях

### Запрос

> Какие ETL-таблицы одновременно являются источниками и приёмниками данных?
> Покажи не более пяти.

### Результат

````text
```table
[["table_name"],["b7000000250002_loansdictstatus::subquery::cr"],["b7000000250006_preferences::subquery::cr"],["b7000000250016_loansmsfoinfo::branch::1"],["b7000000250016_loansmsfoinfo::branch::2"],["b7000000250016_loansmsfoinfo::subquery::msfo1"]]
```
````

## 5. Зависимая цепочка SQLite → Neo4j

### Запрос

> Найди одну ETL-таблицу, которая одновременно выступает источником и
> приёмником данных. Затем покажи её непосредственные входящие и исходящие
> связи.

### Результат

````text
Найдена ETL-таблица `b7000000250002_loansdictstatus::subquery::cr`, которая одновременно выступает источником и приёмником данных.

**Непосредственные связи:**

| direction | source_table | target_table | transformation_ids |
|-----------|--------------|--------------|--------------------|
| upstream  | loansdictstatus | b7000000250002_loansdictstatus::subquery::cr | [14326, 14327, 14328, 14329, 14330] |
| downstream | b7000000250002_loansdictstatus::subquery::cr | b7000000250002_loansdictstatus | [14331, 14332, 14333, 14334] |

**Входящие связи (upstream):**
- Из таблицы `loansdictstatus` поступают данные через 5 трансформаций с ID: 14326, 14327, 14328, 14329, 14330.

**Исходящие связи (downstream):**
- В таблицу `b7000000250002_loansdictstatus` передаются данные через 4 трансформации с ID: 14331, 14332, 14333, 14334.
````

## 6. Два уровня колонкового lineage

### Запрос

> Покажи происхождение данных вниз до двух переходов для колонки
> `s_grnplm_as_t_didsd_305_db_stg.b3050000420007_product.object_id`.
> Перечисли все шаги и отметь переходы через звёздочку.

### Результат

````text
Для колонки `s_grnplm_as_t_didsd_305_db_stg.b3050000420007_product.object_id` найдено два downstream-перехода:

```table
[["step","transformation_id","source_table","target_table","wildcard_passthrough"],["1",17627,"s_grnplm_as_t_didsd_305_db_stg.b3050000420007_product","v_b3050000420007_product_productrate",false],["2",17675,"v_b3050000420007_product_productrate","s305_0007_product_productrate",true]]
```

**Шаги:**  
1. `object_id` из таблицы `s_grnplm_as_t_didsd_305_db_stg.b3050000420007_product` → через трансформацию ID 17627 → в таблицу `v_b3050000420007_product_productrate`.  
2. `object_id` из таблицы `v_b3050000420007_product_productrate` → через трансформацию ID 17675 (wildcard-пропуск) → в таблицу `s305_0007_product_productrate`.
````

## 7. Табличный lineage Greenplum SQL

### Запрос

> Какие физические таблицы участвуют в этом Greenplum SQL и какой объект он
> создаёт? SQL не выполняй:
>
> ```sql
> CREATE VIEW mart.order_customer AS
> SELECT o.id, c.name
> FROM raw.orders o
> JOIN raw.customers c ON c.id = o.customer_id
> ```

### Результат

````text
SQL-запрос создаёт виртуальную таблицу (представление) **`mart.order_customer`**. Физические таблицы, участвующие в запросе: `raw.orders`, `raw.customers`.
````

## 8. Последний файл и его листы

### Запрос

> Какой файл был загружен последним? Перечисли листы этого файла.

### Результат

````text
Последним загруженным файлом является `s2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx`. Он содержит следующие листы:

```table
["a2b_columns","additional_objects","change_log","pxf2a","s2t","s305_0999_enums","source_columns","source_tables","target_columns","target_tables"]
```
````
