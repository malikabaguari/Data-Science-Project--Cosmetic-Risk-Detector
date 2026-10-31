"""Auditable cleaning of the supplied labeled snapshot; never regenerate unknown labels."""
from __future__ import annotations
import ast
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd

REQUIRED = ['code', 'ingredients_list', 'has_risk']
# Curated, exact taxonomy identifiers. Not a legal product classification.
CATEGORY_TAGS = {
    'en:cosmetic-products', 'en:hair', 'en:shampoos', 'en:hair-conditioners',
    'en:hair-care', 'en:hygiene', 'en:showers-and-baths', 'en:shower-gels',
    'en:deodorants', 'en:soaps', 'en:suncare', 'en:in-sun-protections',
    'en:sunscreen', 'en:body', 'en:face', 'en:makeup', 'en:skin-care',
    'en:hand-creams', 'en:body-creams', 'en:facial-creams',
    'en:toothpaste', 'en:toothpastes', 'en:mouthwashes', 'en:lip-balms',
    'en:perfumes', 'en:fragrances', 'en:nail-polishes', 'en:shaving',
}
NON_COSMETIC_TAGS = {
    'en:foods', 'en:food', 'en:food-supplements', 'en:dietary-supplements',
    'en:beverages', 'en:cleaning-products', 'en:detergents', 'en:household-products',
    'en:pet-food', 'en:pet-foods', 'en:laundry-detergents',
}

def canonical_text(value: str) -> str:
    value = html.unescape(unicodedata.normalize('NFKC', str(value))).lower()
    return re.sub(r'\s+', ' ', value).strip()

def parse_ingredients(value) -> list[str]:
    """Only accept real lists or a literal list; never use eval or silently split prose."""
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError):
            return []
    else:
        return []
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        return []
    return [canonical_text(x) for x in parsed if canonical_text(x)]

def formula_group(items: list[str]) -> str:
    # Order-insensitive grouping is conservative: same ingredients, even in a different order,
    # never appear in both training and test. Concentration differences remain unavailable.
    fingerprint = ' | '.join(sorted(set(items)))
    return hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()

def token_overlap(raw, items) -> float:
    a = set(re.findall(r'[a-z]{3,}', canonical_text(raw)))
    b = set(re.findall(r'[a-z]{3,}', ' '.join(items)))
    return len(a & b) / max(1, len(b))

def scope_status(value) -> str:
    tags = {x.strip().lower() for x in str(value).split(',')}
    if tags & NON_COSMETIC_TAGS:
        return 'excluded_category'
    if tags & CATEGORY_TAGS:
        return 'recognized_cosmetic_or_personal_care'
    return 'unclassified'

def clean_data(df: pd.DataFrame):
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    data = df.copy()
    report = {'input_rows': len(data), 'input_columns': len(data.columns)}
    selected = [c for c in ['code','product_name','brands','categories','categories_tags',
                           'categories_fr','main_category_fr','countries_tags',
                           'ingredients_text','ingredients_list','n_ingredients','has_risk']
                if c in data]
    data = data[selected].copy()
    report['selected_source_columns'] = selected
    report['dropped_source_columns'] = len(df.columns) - len(selected)
    data['code'] = data['code'].astype('string')
    data['parsed_ingredients'] = data['ingredients_list'].map(parse_ingredients)
    data['ingredient_count'] = data['parsed_ingredients'].map(len)
    data['label_numeric'] = pd.to_numeric(data['has_risk'], errors='coerce')
    valid = data['ingredient_count'].gt(0) & data['label_numeric'].isin([0,1]) & data['code'].notna()
    report['invalid_rows_removed'] = int((~valid).sum())
    rejected = data.loc[~valid].copy()
    data = data.loc[valid].copy()
    report['duplicate_codes_removed'] = int(data['code'].duplicated().sum())
    data = data.drop_duplicates('code', keep='first').copy()
    data['has_risk'] = data['label_numeric'].astype('int8')
    data['model_text'] = data['parsed_ingredients'].map(lambda x: ', '.join(x))
    data['formula_group'] = data['parsed_ingredients'].map(formula_group)
    data['scope_status'] = data.get('categories_tags', pd.Series('', index=data.index)).map(scope_status)
    data['category_label'] = data.get('main_category_fr', pd.Series('',index=data.index)).fillna('Unknown').astype(str)
    data['raw_list_overlap'] = [token_overlap(r, x) for r,x in zip(data.get('ingredients_text',pd.Series('',index=data.index)).fillna(''),data['parsed_ingredients'])]
    data['low_alignment_flag'] = data['raw_list_overlap'].lt(0.2)
    data['long_prose_flag'] = data['parsed_ingredients'].map(lambda xs: any(len(s)>200 for s in xs))
    q1,q3 = data['ingredient_count'].quantile([.25,.75]); upper=float(q3+1.5*(q3-q1))
    data['long_list_flag'] = data['ingredient_count'].gt(upper)
    report['ingredient_count_iqr_upper_fence'] = upper
    report['long_lists_flagged_not_deleted'] = int(data['long_list_flag'].sum())
    report['low_alignment_flags'] = int(data['low_alignment_flag'].sum())
    report['prose_flags'] = int(data['long_prose_flag'].sum())
    if 'n_ingredients' in data:
        report['count_discrepancies'] = int(pd.to_numeric(data['n_ingredients'],errors='coerce').ne(data['ingredient_count']).sum())
    conflicting = data.groupby('formula_group')['has_risk'].nunique()
    conflict_groups = set(conflicting[conflicting.gt(1)].index)
    report['conflicting_formula_groups_quarantined'] = len(conflict_groups)
    conflict_rows = data['formula_group'].isin(conflict_groups)
    report['conflicting_formula_rows_quarantined'] = int(conflict_rows.sum())
    rejected = pd.concat([rejected, data.loc[conflict_rows]], ignore_index=True)
    # Low source/list overlap is quarantined, not assigned a guessed new label.
    quarantined = conflict_rows | data['low_alignment_flag']
    report['quarantined_total_rows'] = int(quarantined.sum())
    rejected = pd.concat([rejected, data.loc[data['low_alignment_flag'] & ~conflict_rows]],ignore_index=True)
    data = data.loc[~quarantined].drop(columns=['label_numeric']).copy()
    report.update({
        'clean_rows': len(data), 'positive_rows': int(data['has_risk'].sum()),
        'positive_rate': float(data['has_risk'].mean()),
        'unique_formula_groups': data['formula_group'].nunique(),
        'duplicate_formula_rows': int(data['formula_group'].duplicated().sum()),
        'scope_counts': {str(k):int(v) for k,v in data['scope_status'].value_counts().items()},
    })
    return data.reset_index(drop=True), report, rejected

def save_clean(data, report, rejected, root: Path):
    root = Path(root)
    (root/'data/processed').mkdir(parents=True, exist_ok=True)
    (root/'reports/tables').mkdir(parents=True, exist_ok=True)
    out = data.copy()
    out['parsed_ingredients'] = out['parsed_ingredients'].map(json.dumps)
    out.to_csv(root/'data/processed/products_clean.csv',index=False)
    rejected.drop(columns=['parsed_ingredients'],errors='ignore').to_csv(root/'reports/tables/quarantined_rows.csv',index=False)
    (root/'reports/cleaning_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');args=p.parse_args()
    root=Path(args.root);frame=pd.read_csv(root/'data/processed/products_labeled.csv',dtype={'code':'string'},low_memory=False)
    clean,report,rejected=clean_data(frame);save_clean(clean,report,rejected,root)
    print(json.dumps(report,indent=2))
