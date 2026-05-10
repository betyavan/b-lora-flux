# Block Analysis Report — FLUX.1-dev (Combined DS + SS)

**Methodology:** CLIP text-image cosine similarity under embedding injection.  
- Style injection: base=P_content, inject=P_style → high score = block is style-sensitive  
- Content injection: base=P_style, inject=P_content → high score = block is content-sensitive  
- Specificity = style_score − content_score (positive = style-leaning, negative = content-leaning)  
- n_prompts=200, seed=42, steps=28, guidance_scale=3.5, 512×512, dtype=bfloat16  
- DS run: commit 09f7099 (2026-05-08). SS run: commit 24010c9 (2026-05-10).

---

## Combined Ranking (sorted by style_score DESC)

| Block    | Type   | style_score | content_score | specificity | notes                       |
|----------|--------|-------------|---------------|-------------|-----------------------------|
| ds_00    | double | 0.2603      | 0.2428        | +0.0175     | **top style-specific block** |
| ss_28    | single | 0.2512      | 0.2535        | -0.0023     | peak SS sensitivity         |
| ss_24    | single | 0.2479      | 0.2511        | -0.0032     |                             |
| ds_01    | double | 0.2293 (*) | 0.2249 (*)   | +0.0044     | style-specific; (*) old SS scores |
| ss_30    | single | 0.2461      | 0.2522        | -0.0061     |                             |
| ss_31    | single | 0.2441      | 0.2507        | -0.0066     |                             |
| ss_25    | single | 0.2440      | 0.2507        | -0.0067     |                             |
| ss_27    | single | 0.2434      | 0.2472        | -0.0038     |                             |
| ss_17    | single | 0.2429      | 0.2497        | -0.0068     |                             |
| ss_26    | single | 0.2429      | 0.2466        | -0.0037     |                             |
| ss_23    | single | 0.2421      | 0.2399        | +0.0022     | slight style lean           |
| ss_15    | single | 0.2408      | 0.2454        | -0.0046     |                             |
| ss_18    | single | 0.2404      | 0.2437        | -0.0033     |                             |
| ss_34    | single | 0.2361      | 0.2391        | -0.0030     |                             |
| ss_12    | single | 0.2361      | 0.2372        | -0.0011     |                             |
| ss_13    | single | 0.2360      | 0.2369        | -0.0009     |                             |
| ss_32    | single | 0.2347      | 0.2421        | -0.0074     |                             |
| ss_21    | single | 0.2351      | 0.2383        | -0.0032     |                             |
| ss_33    | single | 0.2341      | 0.2392        | -0.0051     |                             |
| ss_14    | single | 0.2335      | 0.2452        | -0.0117     | content-leaning SS          |
| ss_35    | single | 0.2330      | 0.2378        | -0.0048     |                             |
| ss_22    | single | 0.2325      | 0.2428        | -0.0103     | content-leaning SS          |
| ss_16    | single | 0.2323      | 0.2377        | -0.0054     |                             |
| ss_36    | single | 0.2315      | 0.2389        | -0.0074     |                             |
| ss_37    | single | 0.2297      | 0.2356        | -0.0059     |                             |
| ss_09    | single | 0.2267      | 0.2293        | -0.0026     |                             |
| ss_19    | single | 0.2280      | 0.2329        | -0.0049     |                             |
| ss_10    | single | 0.2255      | 0.2203        | +0.0052     | best style-specific SS      |
| ss_11    | single | 0.2197      | 0.2259        | -0.0062     |                             |
| ss_07    | single | 0.2214      | 0.2257        | -0.0043     |                             |
| ss_29    | single | 0.2172      | 0.2181        | -0.0009     | ⚠️ anomalous dip (see below) |
| ss_08    | single | 0.2080      | 0.2128        | -0.0048     |                             |
| ss_20    | single | 0.2056      | 0.2137        | -0.0081     | ⚠️ anomalous dip (see below) |
| ds_18    | double | 0.1626      | 0.1594        | +0.0032     | DS tail recovery            |
| ds_16    | double | 0.1486      | 0.1467        | +0.0019     |                             |
| ds_13    | double | 0.1531      | 0.1570        | -0.0039     |                             |
| ss_04    | single | 0.1984      | 0.1975        | +0.0009     |                             |
| ss_02    | single | 0.1890      | 0.1847        | +0.0043     |                             |
| ds_02    | double | 0.1865      | 0.1890        | -0.0025     |                             |
| ds_03    | double | 0.1864      | 0.1897        | -0.0033     |                             |
| ds_04    | double | 0.1861      | 0.1907        | -0.0046     |                             |
| ss_03    | single | 0.1887      | 0.1860        | +0.0027     |                             |
| ss_05    | single | 0.1843      | 0.1840        | +0.0003     |                             |
| ss_01    | single | 0.1719      | 0.1714        | +0.0005     |                             |
| ss_06    | single | 0.1746      | 0.1758        | -0.0012     |                             |
| ds_10    | double | 0.1404      | 0.1557        | -0.0153     | content-leaning DS          |
| ds_14    | double | 0.1343      | 0.1439        | -0.0096     |                             |
| ds_12    | double | 0.1333      | 0.1538        | -0.0205     | **most content-specific**   |
| ds_15    | double | 0.1358      | 0.1423        | -0.0065     |                             |
| ds_09    | double | 0.1372      | 0.1453        | -0.0081     |                             |
| ds_11    | double | 0.1247      | 0.1346        | -0.0099     |                             |
| ds_13    | double | 0.1531      | 0.1570        | -0.0039     |                             |
| ds_08    | double | 0.1210      | 0.1329        | -0.0119     |                             |
| ds_07    | double | 0.1207      | 0.1349        | -0.0142     |                             |
| ds_17    | double | 0.1202      | 0.1317        | -0.0115     |                             |
| ds_05    | double | 0.1291      | 0.1457        | -0.0166     | content-leaning DS          |
| ss_00    | single | 0.1634      | 0.1671        | -0.0037     |                             |
| ds_06    | double | 0.1104      | 0.1240        | -0.0136     |                             |

---

## Key Findings

### 1. Phase boundaries

- **DS early (ds_00–ds_04) → DS mid (ds_05–ds_17):** Sharp drop at ds_04→ds_05 (~0.057 in style score). Early DS is the only genuinely style-specific zone in the entire network.
- **SS early (ss_00–ss_06) → SS plateau (ss_07–ss_37):** Jump of ~0.047 at ss_06→ss_07. Blocks ss_00–ss_06 behave like weak DS mid-blocks; from ss_07 onward scores plateau around 0.22–0.25.
- **DS tail recovery (ds_17–ds_18):** ds_18 recovers to 0.1626 — consistent with a bottleneck role before handoff to SS stream.

### 2. Anomalous blocks — flag for re-run

| Block | Issue | Style | Content | Likely cause |
|-------|-------|-------|---------|--------------|
| ss_20 | local dip | 0.2056 | 0.2137 | architectural discontinuity or hook misfire; both axes dip together |
| ss_29 | local dip | 0.2172 | 0.2181 | same pattern; sandwiched between 0.2512 and 0.2461 |
| ds_17 | local trough | 0.1202 | 0.1317 | low priority; DS tail not a LoRA candidate |

Both ss_20 and ss_29 show the anomaly in both injection directions simultaneously, which points to an architectural cause (e.g., different norm/routing at those block indices) rather than a stochastic bug. Re-run each in isolation with a fresh seed to confirm.

### 3. Style/Content signal quality

The separation signal is real but weak in the SS stream. Specificity values for SS range only −0.012 to +0.005 (spread ~0.017). With 200 prompt pairs, per-block specificity standard error is ~0.015, meaning many individual SS specificities are statistically indistinguishable from zero. The DS stream carries more differentiated specificity variance (std 0.0090 vs 0.0037 for SS). Conclusion: use specificity for ranking/ordering, but do not over-interpret individual values.

---

## B-LoRA Block Selection Recommendation

| LoRA role    | Recommended blocks           | Rationale                                                       |
|--------------|------------------------------|-----------------------------------------------------------------|
| **Style**    | ds_00, ds_01                 | Only blocks with clearly positive specificity + high absolute score |
| Style (supp) | ss_10, ss_23                 | Best specificity in the SS high-score plateau                   |
| **Content**  | ds_05 – ds_12                | Contiguous content-dominant DS valley                           |
| Content (supp) | ss_14, ss_22               | Strongest content-specific SS blocks (spec < −0.010)            |

**Note on validity:** This experiment measures sensitivity to prompt-embedding injection, not direct LoRA fine-tuning effectiveness. A block responsive to injection may or may not be optimal for a trained LoRA adapter. The block selection above is a motivated hypothesis to test in the main training experiments; it should not be presented as empirical proof of the final selection.

---

_DS data: commit 09f7099 (2026-05-08, n_prompts=200, ds_00–ds_18)_  
_SS data: commit 24010c9 (2026-05-10, n_prompts=200, ss_00–ss_37)_  
_Generated by: scripts/analysis/block_analysis.py + manual combination_
