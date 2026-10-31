-- Execute against data/processed/cosmetic_products.sqlite after running the notebook.
-- Labels are supplied ingredient-based proxies, not safety outcomes.

-- 1. Overall sample size and supplied-label prevalence.
SELECT COUNT(*) AS total_products,
       SUM(has_risk) AS supplied_alerts,
       AVG(CAST(has_risk AS FLOAT)) AS positive_rate
FROM products;

-- 2. Minimum-size category descriptive analysis.
SELECT category_label, COUNT(*) AS products,
       AVG(CAST(has_risk AS FLOAT)) AS positive_rate
FROM products
GROUP BY category_label
HAVING COUNT(*) >= 50
ORDER BY products DESC;

-- 3. Ingredient count and source/list alignment.
SELECT AVG(ingredient_count) AS mean_ingredients,
       MAX(ingredient_count) AS max_ingredients,
       SUM(long_list_flag) AS long_lists_flagged,
       SUM(long_prose_flag) AS prose_flags
FROM products;

-- 4. Repeated normalized formula groups.
SELECT formula_group, COUNT(*) AS product_rows
FROM products
GROUP BY formula_group
HAVING COUNT(*) > 1
ORDER BY product_rows DESC;

-- 5. Operational category scope, not legal classification.
SELECT scope_status, COUNT(*) AS products,
       AVG(CAST(has_risk AS FLOAT)) AS positive_rate
FROM products
GROUP BY scope_status;
