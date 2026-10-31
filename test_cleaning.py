import json
import pandas as pd
import pytest
from src.cleaning import parse_ingredients,formula_group,scope_status,clean_data
from src.deterministic import load_registry,match_ingredients
from src.predict import predict_text

def test_literal_parsing_and_normalization():
    assert parse_ingredients("['AQUA', ' GLYCERIN  ']")==['aqua','glycerin']

def test_not_eval():
    assert parse_ingredients("__import__('os').system('echo unsafe')")==[]

def test_not_prose_split():
    assert parse_ingredients('Aqua, Glycerin')==[]

def test_order_independent_formula_group():
    assert formula_group(['aqua','glycerin'])==formula_group(['glycerin','aqua','aqua'])

def test_empty_input_rejected():
    with pytest.raises(ValueError): predict_text(None,'')

def test_toothpaste_scope():
    assert scope_status('en:hygiene,en:toothpastes')=='recognized_cosmetic_or_personal_care'

def test_missing_registry_fails_explicitly(tmp_path):
    p=tmp_path/'registry.json';p.write_text(json.dumps({'substances':[]}))
    with pytest.raises(ValueError,match='unavailable'):load_registry(p)

def test_exact_match_not_substring():
    registry=[{'canonical_name':'demo term','source_url':'https://example.com','synonyms':[]}]
    assert match_ingredients(['demo term'],registry)['has_risk']==1
    assert match_ingredients(['not demo term'],registry)['has_risk']==0

def test_conflicting_formulas_quarantined():
    frame=pd.DataFrame({'code':['1','2'],'ingredients_list':["['Aqua']","['Aqua']"],
                        'ingredients_text':['Aqua','Aqua'],'has_risk':[0,1]})
    clean,report,_=clean_data(frame)
    assert clean.empty
    assert report['conflicting_formula_groups_quarantined']==1
