#!/usr/bin/env python3
"""
개선된 프롬프트 (영어, chat.completions 방식에 맞춤)
"""

def get_system_prompt():
    return """You are an expert data extractor for biomedical ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity). 
You read main text, figures, and supplementary materials, and return STRICT JSON only. 
No explanations, no markdown, no code blocks - ONLY valid JSON."""

def get_user_prompt(text_content, img_summary, supp_summary, coref_summary, pmc_id):
    return f"""## Task (Objective)
Extract **ALL compounds** and their **complete attributes/indicators** from the provided data sources by **organically connecting**:
- Main text (ADMET-related sentences)
- Figures/Tables (from main paper image analysis)
- Supplementary materials (Excel, Word, PDF text, PDF images)
- Coreference dictionary (entity aliases and relationships)

Output MUST be **STRICT JSON** following the exact schema. No markdown, no explanations.

## Data Sources (Provided Below)
You have access to:
1. **Main Text**: ADMET-related sentences extracted from PDF
2. **Image Analysis**: Figures and tables from main paper with compound information
3. **Supplementary Materials**: 
   - Excel data (compound-attribute pairs)
   - Word data (compound-attribute pairs)
   - PDF text (preview)
   - PDF images (YOLO-extracted compound data: SMILES, Well Position, etc.)
4. **Coreference Dictionary**: Compound aliases and relationships

**CRITICAL - Exhaustiveness:**
- **NEVER stop after the first match**. Extract from ALL pages, ALL tables, ALL sheets, ALL figures.
- Process EVERY compound found across ALL sources.
- Cross-reference between sources to prevent omissions.

## Data Provided:

**Main Text (ADMET-related only):**
{text_content}

{img_summary}

{supp_summary}

{coref_summary}

## What to Extract

For EACH compound, extract:

### Identifiers
- `compound_name`: Canonical name (use coreference to resolve aliases)
- `aliases`: List of all aliases/abbreviations (e.g., ["APA", "Liposomal APA"])
- `smiles`: SMILES string if available, else null
- `inchi`: InChI if available, else null
- `well_position`: Plate position if available (e.g., "1 A01"), else null
- `source_ids`: List of sources where found (e.g., ["Supplementary Table 1", "Figure 2", "Main text p.14"])

### ADMET Standard Indicators (22 + any additional found)
1. **caco2**: Caco-2 Permeability (units: 1e-6 cm/s or Papp)
2. **mdck**: MDCK Permeability (units: 1e-6 cm/s or Papp)
3. **pampa**: PAMPA (units: 1e-6 cm/s)
4. **lipinski**: Lipinski's Rule of Five (rule_of_five_pass, molecular_weight, logp, hbd, hba)
5. **logd**: Distribution coefficient (ph-dependent, include ph if available)
6. **logs**: Aqueous solubility (log units)
7. **pka**: Acid dissociation constant
8. **ppb**: Plasma protein binding (units: %)
9. **bbb**: Blood-brain barrier (units: logBB)
10. **vd**: Volume of distribution (units: L/kg)
11. **cyp1a2**: CYP1A2 inhibition (units: µM, assay: IC50)
12. **cyp2c9**: CYP2C9 inhibition (units: µM, assay: IC50)
13. **cyp2c19**: CYP2C19 inhibition (units: µM, assay: IC50)
14. **cyp2d6**: CYP2D6 inhibition (units: µM, assay: IC50)
15. **cyp3a4**: CYP3A4 inhibition (units: µM, assay: IC50)
16. **cyp_inhibition**: General CYP inhibition status (yes/no/unknown)
17. **cl**: Clearance (units: mL/min/kg or L/h)
18. **t_half**: Half-life (units: hours or h)
19. **herg**: hERG IC50 (units: µM, assay: IC50)
20. **dili**: Drug-induced liver injury risk (High/Medium/Low)
21. **ames**: Ames test result (Positive/Negative)
22. **carcinogenicity**: Carcinogenicity result (Positive/Negative)
23. **additional_indicators**: Any other attributes found (EC50, Ki, solubility at pH X, etc.)

### Normalization Rules
- **Alias Merging (Coreference)**: Use coreference dictionary to merge aliases into one record.
  Example: "APA" = "6-(4-aminophenyl)-N-(3,4,5-trimethoxyphenyl)pyrazin-2-amine" = "Liposomal APA" → ONE record with aliases list.
- **SMILES Canonicalization**: Normalize SMILES notation if possible.
- **Unit Normalization**: 
  - µM/μM/uM → "µM"
  - hr/h → "h"
  - 1e-6 cm/s → "1e-6 cm/s" (consistent)
  - Preserve original unit in `raw_unit` if different from normalized
- **Conflict Resolution (Priority)**: supplement > image > text
  - If conflicting values, use supplement data
  - Document conflicts in `notes` field

## Output Schema (STRICT JSON - MUST FOLLOW EXACTLY)

```json
{{
  "schema_version": "2.0",
  "created_at": "<ISO 8601 timestamp>",
  "pmc_id": "{pmc_id}",
  "records": [
    {{
      "compound_name": "<canonical name>",
      "aliases": ["<alias1>", "<alias2>"],
      "smiles": "<SMILES or null>",
      "inchi": "<InChI or null>",
      "well_position": "<e.g., '1 A01' or null>",
      "source_ids": ["<source1>", "<source2>"],
      "caco2": {{"value": <number|null>, "unit": "1e-6 cm/s"|"Papp"|null, "source": "text|supplement|image"}},
      "mdck": {{"value": <number|null>, "unit": "1e-6 cm/s"|"Papp"|null, "source": "text|supplement|image"}},
      "pampa": {{"value": <number|null>, "unit": "1e-6 cm/s"|null, "source": "text|supplement|image"}},
      "lipinski": {{"rule_of_five_pass": true|false|null, "molecular_weight": <number|null>, "logp": <number|null>, "hbd": <number|null>, "hba": <number|null>, "source": "text|supplement|image"}},
      "logd": {{"value": <number|null>, "ph": <number|null>, "source": "text|supplement|image"}},
      "logs": {{"value": <number|null>, "unit": "log units"|null, "source": "text|supplement|image"}},
      "pka": {{"value": <number|null>, "source": "text|supplement|image"}},
      "ppb": {{"value": <number|null>, "unit": "%", "source": "text|supplement|image"}},
      "bbb": {{"value": <number|null>, "unit": "logBB", "source": "text|supplement|image"}},
      "vd": {{"value": <number|null>, "unit": "L/kg", "source": "text|supplement|image"}},
      "cyp1a2": {{"value": <number|null>, "unit": "µM", "assay": "IC50"|null, "source": "text|supplement|image"}},
      "cyp2c9": {{"value": <number|null>, "unit": "µM", "assay": "IC50"|null, "source": "text|supplement|image"}},
      "cyp2c19": {{"value": <number|null>, "unit": "µM", "assay": "IC50"|null, "source": "text|supplement|image"}},
      "cyp2d6": {{"value": <number|null>, "unit": "µM", "assay": "IC50"|null, "source": "text|supplement|image"}},
      "cyp3a4": {{"value": <number|null>, "unit": "µM", "assay": "IC50"|null, "source": "text|supplement|image"}},
      "cyp_inhibition": {{"status": "yes|no|unknown"|null, "source": "text|supplement|image"}},
      "cl": {{"value": <number|null>, "unit": "mL/min/kg"|"L/h"|null, "source": "text|supplement|image"}},
      "t_half": {{"value": <number|null>, "unit": "h", "source": "text|supplement|image"}},
      "herg": {{"value": <number|null>, "unit": "µM", "assay": "IC50"|null, "source": "text|supplement|image"}},
      "dili": {{"risk": "High|Medium|Low"|null, "source": "text|supplement|image"}},
      "ames": {{"result": "Positive|Negative"|null, "source": "text|supplement|image"}},
      "carcinogenicity": {{"result": "Positive|Negative"|null, "source": "text|supplement|image"}},
      "additional_indicators": {{
        "<indicator_name>": {{"value": <any>, "unit": "<string|null>", "source": "text|supplement|image"}}
      }},
      "notes": "<conflicts/assumptions or null>"
    }}
  ]
}}
```

## Procedure (Steps)
1. **Collect all compound candidates** from all sources (text, images, supplements)
2. **For each compound**, extract ALL attributes/indicators from ALL sources
3. **Use coreference dictionary** to merge aliases into canonical names
4. **Normalize units and notation** (SMILES, units, etc.)
5. **Resolve conflicts** using priority (supplement > image > text), document in `notes`
6. **Output STRICT JSON only** - no markdown, no explanations, no code blocks

## Exhaustiveness (Critical)
- Extract **ALL compounds** found, not just a subset
- Include compounds even if ADMET data is incomplete (use null for missing values)
- Cross-reference between sources to ensure no omissions
- Process supplementary Excel/Word data completely (all rows, all sheets)
- Process all YOLO-extracted compounds from PDF images

## Constraints
- Output MUST be **valid JSON object only**
- **NO markdown formatting** (no ```json code blocks)
- **NO explanations** or natural language
- **NO code** or comments
- If a field is missing, use `null` but **keep the field**

## Final Reminder
Return ONLY the JSON object. Nothing else.""".format(
        text_content=text_content,
        img_summary=img_summary,
        supp_summary=supp_summary,
        coref_summary=coref_summary,
        pmc_id=pmc_id
    )



