"""Auditable exact matching once an actual documented registry is supplied."""
import json
from pathlib import Path
from .cleaning import canonical_text

def load_registry(path):
    registry=json.loads(Path(path).read_text(encoding='utf-8'))
    substances=registry.get('substances',[])
    if not substances:
        raise ValueError('Exact reference registry unavailable. No deterministic metric can be computed.')
    for row in substances:
        if not row.get('canonical_name') or not row.get('source_url'):
            raise ValueError('Every reference substance needs a canonical name and a source URL.')
    return substances

def match_ingredients(ingredients,registry):
    values={canonical_text(x) for x in ingredients}
    matches=[]
    for row in registry:
        aliases={canonical_text(row['canonical_name'])}|{canonical_text(x) for x in row.get('synonyms',[])}
        if values & aliases:
            matches.append(row['canonical_name'])
    return {'has_risk':int(bool(matches)),'matched_substances':matches}
