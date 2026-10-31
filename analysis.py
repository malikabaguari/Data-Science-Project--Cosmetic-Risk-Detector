"""Descriptive analysis and saved charts; rates describe labels, not product safety."""
from pathlib import Path
from collections import Counter
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

COLORS=['#1a1a1a','#d9281c','#e6d3d3','#777777']

# Display translations only: source fields, group membership and model inputs stay unchanged.
CATEGORY_DISPLAY_EN = {
    'Shampoings': 'Shampoos',
    'Gels douche': 'Shower gels',
    'Déodorants': 'Deodorants',
    'Déodorants bille': 'Roll-on deodorants',
    'Dentifrices': 'Toothpastes',
    'Savons': 'Soaps',
    'Savons-liquides': 'Liquid soaps',
    'Crèmes solaire': 'Sunscreen creams',
    'Crèmes pour les mains': 'Hand creams',
    'Crèmes pour le corps': 'Body creams',
    'Crèmes pour le visage': 'Face creams',
    'Crèmes douche': 'Shower creams',
    'Laits pour le corps': 'Body lotions',
    'Après-shampoings': 'Hair conditioners',
    'Soins cheveux': 'Hair care',
    'Shampoings anti-pelliculaires': 'Anti-dandruff shampoos',
    'Shampoings pour cheveux secs': 'Shampoos for dry hair',
    'en:cosmetic-products': 'Cosmetic products',
    'en:open-beauty-facts': 'Open Beauty Facts (source tag)',
    'en:toothpaste': 'Toothpaste (source tag)',
    'nl:verzorging': 'Personal care (source tag)',
}

def english_category_label(value):
    """Translate known presentation labels, never rewrite original taxonomy fields."""
    return CATEGORY_DISPLAY_EN.get(value, value)

def save_eda(data, original, root):
    root=Path(root);figdir=root/'reports/figures';tabdir=root/'reports/tables'
    figdir.mkdir(parents=True,exist_ok=True);tabdir.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style='whitegrid',font_scale=1.05)
    def finish(name,title,xlabel='',ylabel=''):
        plt.title(title);plt.xlabel(xlabel);plt.ylabel(ylabel);plt.tight_layout()
        plt.savefig(figdir/f'{name}.png',dpi=150);plt.close()
    missing=original.isna().mean().sort_values(ascending=False)
    missing.to_csv(tabdir/'missing_values.csv',header=['missing_fraction'])
    selected=[c for c in ['product_name','brands','categories_tags','countries_tags','ingredients_text','ingredients_list','has_risk'] if c in original]
    plt.figure(figsize=(9,4));(original[selected].isna().mean()*100).plot.barh(color=COLORS[0])
    finish('01_missing_values','Missing values in relevant source fields','Missing (%)')
    counts=data['has_risk'].value_counts().sort_index();plt.figure(figsize=(7,4))
    plt.bar(['No supplied alert (0)','Supplied alert (1)'],counts,color=COLORS[:2]);
    for i,n in enumerate(counts):plt.text(i,n+50,f'{n:,}',ha='center')
    finish('02_target_distribution','Label distribution after cleaning',ylabel='Products')
    plt.figure(figsize=(9,4));sns.histplot(data=data,x='ingredient_count',hue='has_risk',bins=40,multiple='layer',palette=COLORS[:2],alpha=.5)
    finish('03_ingredient_count','Ingredient-count distribution; long lists retained',ylabel='Products')
    plt.figure(figsize=(7,4));sns.boxplot(data=data,x='has_risk',y='ingredient_count',color=COLORS[2],showfliers=False)
    finish('04_ingredients_by_label','Ingredient count by supplied label (outlier dots omitted)','Supplied label','Parsed ingredients')
    categories=data.groupby('category_label').agg(products=('has_risk','size'),positive_rate=('has_risk','mean')).sort_values('products',ascending=False)
    categories.to_csv(tabdir/'category_summary.csv')
    plt.figure(figsize=(10,6));categories.head(12)['products'].sort_values().rename(index=english_category_label).plot.barh(color=COLORS[0]);finish('05_top_categories','Top categories (English display labels; original source groups)',xlabel='Products')
    eligible=categories[categories['products']>=50].sort_values('positive_rate',ascending=False).head(12)
    plt.figure(figsize=(10,6));(eligible['positive_rate'].sort_values()*100).rename(index=english_category_label).plot.barh(color=COLORS[1]);finish('06_label_rate_by_category','Supplied-label rate by category (n ≥ 50); not a safety ranking','Positive-label rate (%)')
    ingredient_freq=Counter(ingredient for row in data['parsed_ingredients'] for ingredient in set(row))
    top=pd.Series(dict(ingredient_freq.most_common(20)),name='products');top.to_csv(tabdir/'top_ingredients.csv')
    plt.figure(figsize=(10,6));top.head(12).sort_values().plot.barh(color=COLORS[0]);finish('07_frequent_ingredients','Most frequent parsed ingredient strings','Products containing string')
    # Country tags can be multi-valued; count products in each tag, not mutually exclusive shares.
    country=Counter(t.strip() for s in data['countries_tags'].fillna('') for t in str(s).split(',') if t.strip())
    pd.Series(country,name='products').sort_values(ascending=False).to_csv(tabdir/'country_summary.csv')
    plt.figure(figsize=(9,5));pd.Series(dict(country.most_common(10))).sort_values().plot.barh(color=COLORS[0]);finish('08_country_tags','Market/country tags: French export does not mean France-only','Products')
    numeric=data[['ingredient_count','has_risk','raw_list_overlap']].corr()
    plt.figure(figsize=(7,4));sns.heatmap(numeric,annot=True,fmt='.2f',cmap='RdBu_r',center=0,vmin=-1,vmax=1);finish('09_numeric_associations','Descriptive correlations (not causal)')
    sample=data.drop(columns=['parsed_ingredients']).copy()
    with sqlite3.connect(root/'data/processed/cosmetic_products.sqlite') as con:
        sample.to_sql('products',con,index=False,if_exists='replace')
        query='''SELECT category_label, COUNT(*) AS products, AVG(has_risk) AS positive_rate
                 FROM products GROUP BY category_label HAVING COUNT(*) >= 50
                 ORDER BY products DESC'''
        pd.read_sql_query(query,con).to_csv(tabdir/'sql_category_results.csv',index=False)
    summary={'mean_ingredient_count':float(data['ingredient_count'].mean()),
             'median_ingredient_count':float(data['ingredient_count'].median()),
             'max_ingredient_count':int(data['ingredient_count'].max()),
             'missing_relevant_fields':{c:float(original[c].isna().mean()) for c in selected},
             'top_5_ingredient_strings':dict(ingredient_freq.most_common(5)),
             'country_tag_counts_are_not_exclusive':True}
    (root/'reports/eda_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary,categories,top
