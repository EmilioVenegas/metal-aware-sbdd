# Step 2 Arm C Evaluation: Metal-Aware Pocket Representation Analysis

**Date:** 2026-08-19  
**Evaluation Target:** DiffSBDD with Metal-Aware Pocket Representation (Arm C: `checkpoints/arm_c_best.ckpt`, 16-element vocabulary + LoRA)  
**Pre-Registration Plan:** `results/step2/ANALYSIS_PLAN_ARMC.md`  
**Dataset & Cohort:** 133 external catalytic Zn targets ($N=100$ valid molecules/target, 13,300 total generated molecules; 12,700 in primary X-ray cohort across $m=21$ clusters).  

---

## 1. Executive Summary & Core Decision Rule Verdict

| Metric | Arm A (Status Quo) | Arm B (Data Baseline) | **Arm C (Metal-Aware)** | Native Ceiling (C1) | Arm C Pre-Registered Prediction | Empirical Verdict |
|---|---|---|---|---|---|---|
| **Valid Coordination Rate** | 19.98% | 10.58% | **24.05%** | 77.17% | **> 35.0%** | **NOT MET** (24.05% vs >35%) |
| **Primary Violation ($V1 \lor V2$)** | 18.38% | 11.39% | **25.74%** | 20.47% | **< 15.0%** | **NOT MET** (25.74% vs <15%) |
| **V2-Strict (Chelate-Aware)** | 14.80% | 9.69% | **20.80%** | 2.36% | — | Diagnostic |
| **V1 Hard Clash (< 1.70 Å)** | 7.38% | 2.81% | **11.59%** | 0.00% | — | Elevated clash (+4.21 pp vs A) |
| **V2 Shell Occupancy (< 2.70 Å)** | 15.71% | 10.39% | **21.81%** | 20.47% | — | Increased density (+6.10 pp vs A) |
| **Mean Coordination Count** | 0.215 | 0.110 | **0.260** | 0.874 | **> 0.70** | **NOT MET** (0.260 vs >0.70) |
| **Angular RMSD to Ideal (All)** | 25.28° | 26.90° | **25.02°** | 18.04° | **< 20.0°** | **NOT MET** (25.02° vs <20°) |
| **Angular RMSD (Conditional $\ge 1$)** | 16.67° | 16.12° | **17.94°** | 14.94° | — | Diagnostic (17.94°) |
| **A $\rightarrow$ C1 Gap Closed** | 0.0% | -16.4% | **+7.1%** | 100.0% | > 25.0% | Partial (+7.1%) |

### **Pre-Registered Decision Rule Assessment (§6 of ANALYSIS_PLAN_ARMC.md):**
- **Decision Rule 1 (Core Hypothesis):** *"If Arm C's valid-coordination rate exceeds Arm B's by a wide margin and clears the >35% threshold, the representation-bottleneck hypothesis is supported."*
  - **Verdict:** **REPRESENTATION BOTTLENECK FIX ALONE (UNDER CURRENT LORA/SCALE SETTING) IS INSUFFICIENT TO REACH >35%.**
  - Arm C achieves **24.05%** valid coordination (Cluster bootstrap mean: **28.33%**, 95% CI: [23.42%, 33.06%]).
  - While Arm C significantly outperforms Arm B (+13.47 pp, GEE OR = 2.46, $p = 1.21e-23$) and modestly outperforms Arm A (+4.07 pp, GEE OR = 1.24, $p = 1.72e-06$), it falls well short of the pre-registered >35.0% threshold.
---

## 2. Integrity and Sampling Denominators

- **Cohort Completion:** 133/133 targets reached `complete` status with exactly 100 valid molecules generated per target ($N=13,300$).
- **Validity Rate across Targets:** Mean = 97.70%, Min = 92.50%.
- **Amendment 4 Correlation Check (Validity vs Primary Violation Rate):**
  - Pearson $r = 0.1663$ ($p = 0.06174$)
  - Spearman $\rho = 0.2215$ ($p = 0.01234$)
  - **Promotion rule ($r < -0.30, p < 0.05$):** **NOT TRIGGERED**. Headline analysis remains on the valid-only denominator.

---

## 3. Statistical Contrasts & Hypothesis Testing

### 3.1 Arm C vs. Arm A (Metal-Aware vs Status Quo)
- **Valid Coordination Rate:**
  - Pooled: **24.05%** vs 19.98% ($\Delta = +4.07\%$)
  - Paired Cluster Difference $\bar{D}$: **+1.16%** (95% CI: [-2.01%, 4.25%], $SE = 1.60\%$)
  - GEE Logistic Regression: Odds Ratio = **1.2424** (95% CI: [1.1367, 1.3580], $p = 1.721e-06$)
- **Primary Violation ($V1 \lor V2$):**
  - Pooled: **25.74%** vs 18.38% ($\Delta = +7.36\%$)
  - Paired Cluster Difference $\bar{D}$: **+5.60%** (95% CI: [-0.75%, 10.41%])
  - GEE Logistic Regression: Odds Ratio = **1.4092** (95% CI: [1.1735, 1.6922], $p = 0.0002397$)

### 3.2 Arm C vs. Arm B (Metal-Aware vs Data-Fine-Tuned Metal-Blind)
- **Valid Coordination Rate:**
  - Pooled: **24.05%** vs 10.58% ($\Delta = +13.47\%$)
  - Paired Cluster Difference $\bar{D}$: **+14.55%** (95% CI: [10.63%, 18.45%])
  - GEE Logistic Regression: Odds Ratio = **2.4560** (95% CI: [2.0602, 2.9277], $p = 1.209e-23$)
- **Primary Violation ($V1 \lor V2$):**
  - Paired Cluster Difference $\bar{D}$: **+20.69%** (95% CI: [14.74%, 26.59%])
  - GEE Logistic Regression: Odds Ratio = **2.1963** (95% CI: [1.6016, 3.0117], $p = 1.041e-06$)

### 3.3 Arm C vs. Native Ceiling (C1)
- **Valid Coordination Rate:**
  - Arm C (24.05%) vs Native (77.17%)
  - Paired Cluster Difference $\bar{D}$: **-54.52%** (95% CI: [-69.37%, -36.59%])
  - GEE Logistic Regression: Odds Ratio = **0.0933** (95% CI: [0.0355, 0.2452], $p = 1.495e-06$)
  - **A $\rightarrow$ C1 Gap Closed:** **+7.12%**

---

## 4. Controlled Comparisons (C2 & C3)

### Control C2: Protein-Atom Clash (Paired within Molecule)
- **Hard Clash (< 1.70 Å):**
  - Average Pocket Protein Atom Clash Rate: **0.080%**
  - Metal Site Clash Rate: **11.590%**
  - Paired Difference (Metal − Protein Atom): **+20.027%** (95% CI: [14.664%, 25.735%])
- **Shell Proximity (< 2.70 Å):**
  - Average Pocket Protein Atom Proximity Rate: **0.840%**
  - Metal Site Proximity Rate: **59.200%**
  - Paired Difference (Metal − Protein Atom): **+74.934%** (95% CI: [64.853%, 83.272%])

### Control C3: Burial-Matched Decoys (Paired within Pocket)
- **Metal Site Occupancy ($d \le 2.70$ Å):** **59.20%**
- **Decoy Points Occupancy ($d \le 2.70$ Å):** **41.83%**
- **Occupancy Ratio (Metal / Decoy):** **1.415×**
- **Paired Difference $\bar{D}$ (Metal − Decoy):** **15.49%** (95% CI: [3.73%, 27.97%], $\sigma_d = 0.2881$)
- **Post-Hoc MDE (80% Power):** **17.60%**

---

## 5. Mechanistic Diagnostics & Distance Redistribution

### 5.1 Distance Shell Redistribution (Nearest Ligand Heavy Atom to Catalytic Zn)

| Distance Shell (Å) | Arm A | Arm B | **Arm C** | Native C1 | Interpretation |
|---|---|---|---|---|---|
| **< 1.70 (Hard Clash)** | 7.38% | 2.81% | **11.55%** | 0.0% | Clash elevated in Arm C |
| **1.70 – 1.90** | 5.66% | 2.63% | **7.63%** | 23.62% | Sub-optimal donor approach |
| **1.90 – 2.35 (Valid Zn–N/O Window)** | 20.64% | 12.33% | **23.88%** | 73.23% | Moderate enrichment in valid shell |
| **2.35 – 2.70** | 15.45% | 12.21% | **16.12%** | 3.15% | Extended coordination shell |
| **2.70 – 3.50** | 22.38% | 21.11% | **18.24%** | 0.0% | Second coordination sphere |
| **3.50 – 5.00** | 14.96% | 16.84% | **10.91%** | 0.0% | Distant / pocket periphery |
| **> 5.00** | 13.54% | 32.06% | **11.67%** | 0.0% | Arm B avoids metal; Arm C engages |

### 5.2 Contacting Elements Breakdown ($d < 2.70$ Å)
- **Arm A Total Contacts:** 9475 -> `{'O': 6002, 'N': 819, 'C': 2279, 'S': 206, 'P': 79, 'F': 89, 'CL': 1}`
- **Arm B Total Contacts:** 5385 -> `{'O': 3217, 'N': 382, 'BR': 1, 'S': 116, 'C': 1573, 'F': 78, 'P': 16, 'CL': 2}`
- **Arm C Total Contacts:** 12166 -> `{'C': 3205, 'O': 7127, 'N': 1375, 'S': 227, 'CL': 8, 'P': 113, 'I': 1, 'F': 110}`
- **Native Total Contacts:** 222 -> `{'N': 64, 'O': 116, 'S': 14, 'C': 28}`

*Key Mechanistic Observation:* Arm C substantially increases heavy-atom density at the metal site relative to Arm B (total shell contacts: 12166 vs 5385), and nitrogen donors nearly double compared to Arm A. However, because the coordinate update layers were frozen to preserve equivariance and training was constrained to LoRA on feature MLPs over 1,101 complexes, the model learned to *place atoms near the metal* without fine spatial distance calibration (causing simultaneous increases in valid coordination and hard clash).

---

## 6. Pre-Registered De-Risking: SMARTS-Baseline Kill Check

Comparison of raw generation vs post-hoc Zinc-Binding-Group (ZBG) SMARTS filtering on the primary X-ray cohort:

| Method | Total Generated | Retained by ZBG Filter | Valid Coord Rate among Retained | Valid Coord Yield per Generated Mol |
|---|---|---|---|---|
| **Arm A (Status Quo, Unfiltered)** | 12,700 | 12,700 (100.0%) | 19.98% | 19.98% |
| **Arm A + SMARTS Filter** | 12,700 | 4813 (37.9%) | **24.45%** | **9.27%** |
| **Arm B (Data Baseline, Unfiltered)** | 12,700 | 12,700 (100.0%) | 10.58% | 10.58% |
| **Arm B + SMARTS Filter** | 12,700 | 2966 (23.35%) | **14.53%** | **3.39%** |
| **Arm C (Metal-Aware, Unfiltered)** | 12,700 | 12,700 (100.0%) | **24.05%** | **24.05%** |
| **Arm C + SMARTS Filter** | 12,700 | 4100 (32.28%) | **29.32%** | **9.46%** |
| **Native Ceiling (C1)** | 127 | 127 (100.0%) | **77.17%** | **77.17%** |

### **Scientific Takeaway on SMARTS Baseline:**
1. **Rate vs Yield Distinction:** The post-hoc SMARTS filter over Arm A achieves a valid coordination rate of **24.45%**, which matches or slightly exceeds Arm C's raw rate of **24.05%**.
2. **Sampling Efficiency:** However, the SMARTS filter discards ~61.5% of generated molecules, resulting in an effective valid coordination yield of only **9.27%** per generated molecule. Arm C provides a **2.3× higher absolute yield (24.05% vs 9.27%)** at fixed generative sampling cost.
3. Combining Arm C with post-hoc SMARTS filtering reaches **29.32%** valid coordination among retained molecules.

---

## 7. Stratified Subgroup: Cryo-EM Targets ($m=5$ clusters, $n=6$ targets, $N=600$ molecules)

- **Valid Coordination Rate:** **29.33%** (Cluster BS: 28.20%, 95% CI: [15.20%, 45.20%])
- **Primary Violation Rate ($V1 \lor V2$):** **45.00%**
- **V2-Strict:** **38.00%**
- **V1 Hard Clash Rate:** **21.50%**
- **V2 Shell Occupancy Rate:** **43.33%**

---

## 8. Full Cohort Summary ($m=26$ clusters, $n=133$ targets, $N=13,300$ molecules)

- **Valid Coordination Rate:** **24.29%**
- **Primary Violation Rate ($V1 \lor V2$):** **26.61%**
- **V1 Hard Clash Rate:** **12.04%**
- **V2 Shell Occupancy Rate:** **22.78%**
- **Mean Coordination Count:** **0.264**
- **Mean Angular RMSD (All):** **24.94°** (Median: 28.66°)
- **Mean Angular RMSD (Conditional $\ge 1$):** **17.95°** (Median: 17.58°)

---

## 9. Comprehensive Conclusions & Methodological Assessment

1. **Failure to Clear Pre-Registered Primary Target (>35%):**
   Arm C reaches **24.05%** valid coordination on the primary X-ray cohort, falling short of the pre-registered >35.0% prediction. The hypothesis that a minimal LoRA fine-tune (0.7% parameters) on 1,101 metal-containing pockets would immediately resolve coordination chemistry is **rejected**.
2. **Clear Evidence of Mechanistic Engagement:**
   Unlike Arm B (which moved molecules away from the metal, causing valid coordination to drop to 10.58%), Arm C's metal atom type successfully directs atoms toward the metal:
   - Shell occupancy increases from 15.71% (Arm A) to **21.81%** (Arm C).
   - Nitrogen contacts in the coordination shell nearly double.
   - Odds of valid coordination significantly exceed Arm B (OR = 2.46, $p = 1.21e-23$).
3. **The Distance Calibration Dilemma:**
   Because LoRA was applied strictly to node feature MLPs while the coordinate/spatial update layers were kept frozen to preserve equivariance, the network learned *what* to place near the metal without learning *exact spatial repulsion/distance tolerances*. Consequently, hard clashes (<1.70 Å) increased from 7.38% (Arm A) to **11.59%** (Arm C), offsetting much of the valid coordination gain.
4. **Takeaway for Architecture and Representation Design:**
   Explicit pocket metal representation is necessary (as shown by Arm B's total failure), but parameter-efficient fine-tuning on feature embeddings alone is insufficient for fine geometric coordination. Full end-to-end spatial conditioning or coordinate-level adaptation is required to achieve native-like metal coordination.
