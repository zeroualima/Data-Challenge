# 🏆 Data Battle IA PAU 2026 — Probabilistic Prediction of Lightning Alert Ends

## 👥 Team

* **Team Name:** L2ay2ay
* **Members:**
* ZEROUALI Mohammed Amine
* BOUBEKRI Saad



---

## 🎯 Problem Statement

Airports must suspend certain activities as soon as a lightning strike occurs within a 20 km radius. With the current method, the alert remains active for a **fixed 30 minutes** after the last observed strike — regardless of the actual dynamics of the thunderstorm.

The objective is to **reduce this waiting time** by estimating, at any given moment, the probability that the alert is still active — allowing for a faster resumption of activities while maintaining a controlled risk level (R < 2%).

---

## 💡 Proposed Solution

Our solution combines two complementary models.

**Step 1 — Hawkes Process (Probabilistic Model)**

Lightning strikes within the 20 km disk are modeled as a self-exciting point process. The conditional intensity `λ*(t) = μ + α Σ exp(-β(t − τᵢ))` is estimated by maximum likelihood (Ogata's method), using a grid of initializations to avoid local minima. Parameters are adjusted **per airport** over 10 years of training data:

| Airport | μ | α | β | α/β |
| --- | --- | --- | --- | --- |
| Ajaccio | 0.0557 | 0.2122 | 0.2411 | 0.880 |
| Bastia | 0.0561 | 0.2146 | 0.2374 | 0.904 |
| Biarritz | 0.0540 | 0.2309 | 0.2527 | 0.914 |
| Nantes | 0.0574 | 0.2310 | 0.2558 | 0.903 |
| Pisa | 0.0562 | 0.2041 | 0.2288 | 0.892 |

This model provides the indicator `λ*(τₙ)` at any given moment — the current excitation level of the thunderstorm — which is used as a key feature for the prediction.

**Step 2 — LightGBM with Quantile Regression (Predictive Model)**

Using 12 temporal features (inter-arrival times, trends, amplitude, `λ*(τₙ)`), 13 LightGBM models predict the 5% to 95% quantiles of the delay before the next strike. These quantiles build a **conditional survival curve**:

```
S(u | Hₙ) = P(T > τₙ + u | Hₙ)
u* = inf{ u ∈ [0, 30] | S(u) < ε }

```

The parameter ε is controlled via the confidence threshold θ = 1 − ε, allowing us to find the best gain/risk trade-off according to the jury's criteria.

**Results on the Evaluation Set (612 alerts, 5 airports):**

| Metric | Value |
| --- | --- |
| Total gain (θ = 0.85) | **80 hours** |
| Risk R | **0.0182 < 0.02** ✅ |
| Covered alerts | 412 / 612 |
| Training (92,377 examples) | < 5 min CPU |

Breakdown per airport at θ = 0.85:

| Airport | Alerts | Gain (h) | Missed |
| --- | --- | --- | --- |
| Ajaccio | 96 | 25.5 | 2 |
| Bastia | 141 | 20.3 | 5 |
| Biarritz | 118 | 7.8 | 0 |
| Nantes | 45 | 5.7 | 0 |
| Pisa | 146 | 21.3 | 0 |

---

## ⚙️ Technical Stack

* **Languages:** Python 3.12
* **Frameworks:** LightGBM, NumPy, pandas, SciPy
* **Tools:** Git, Jupyter (evaluation), matplotlib
* **AI:** LightGBM (gradient boosting, quantile regression) + Hawkes Process (Ogata's MLE)
* **Infrastructure:** 100% local, no cloud or GPU dependencies

---

## 🚀 Installation & Execution

### Prerequisites

* Python 3.10+
* pip

### Installation

```bash
git clone <repository-url>
cd "DATA CHALLENGE"
pip install pandas numpy scipy lightgbm scikit-learn matplotlib tqdm

```

The truncated data provided by the jury is already present in `dataset_test/dataset_set.csv`, meaning you only need to swap it out to generate other predictions.

### Execution

**Complete pipeline (training on all data + generating predictions):**

```bash
python3 -m src.generate_predictions

```

This generates `dataset_test/predictions.csv` for the data in `dataset_test/dataset_set.csv` following the format expected by the jury (columns: `airport`, `airport_alert_id`, `prediction_date`, `predicted_date_end_alert`, `confidence`).

**Local evaluation of predictions:**

```bash
python3 evaluate_predictions.py \
  --data  ./dataset_test/segment_alerts_all_airports_eval.csv \
  --preds ./dataset_test/predictions.csv \
  --output ./figures/evaluation_curve.png

```

---

## 📁 Project Structure

```
DATA CHALLENGE/
├── data_train_databattle2026/
│   └── segment_alerts_all_airports_train.csv    # training data (10 years)
├── dataset_test/
│   ├── dataset_set.csv                          # truncated test data
│   ├── predictions.csv                          # generated predictions (submission)
│   └── Evaluation_databattle_meteorage.ipynb    # jury notebook
├── src/
│   ├── preprocessing.py   # alert segmentation
│   ├── hawkes.py          # Hawkes process MLE estimation
│   ├── features.py        # feature engineering
│   ├── lgbm_model.py      # LightGBM training by quantile + u*
│   ├── simulation.py      # validation via simulation (Ogata's thinning)
│   ├── validation.py      # goodness-of-fit
│   └── generate_predictions.py   # complete pipeline → predictions.csv
├── main.py                        # training + local evaluation (train only)
├── evaluate_predictions.py        # local evaluation (gain/risk Pareto curve)
├── Rapport/
│   └── rapport_final.pdf         # complete report
├── Presentation/
|   └── presentation.pdf
└── figures/                       # generated graphs

```

---

## 📊 Detailed Model and Results

The `Rapport/Data_Challenge.pdf` file contains:

* The rigorous mathematical formulation of the problem
* The derivation of the conditional survival curve
* The quantile calibration results
* The LightGBM feature importance
* An analysis of the stationary Hawkes model limitations

### Most Important Features (LightGBM Gain)

| Feature | Description |
| --- | --- |
| `ia_max` | Maximum inter-arrival time observed in the alert |
| `ia_min` | Minimum inter-arrival time observed in the alert |
| `duree_ecoulee` | Time elapsed since the start of the alert |
| `lambda_hawkes` | Current Hawkes intensity λ*(τₙ) |
| `ia_last` | Last observed inter-arrival time |

---

## 🌱 Environmental and Social Impact

* No cloud or GPU dependency — runs entirely on a standard laptop
* Training on 92,377 examples in under 5 minutes of CPU time
* Vectorized inference: 37,000 predictions in under 30 seconds
* Estimated gain of **80 hours** across ~412 test alerts → faster resumption of airport operations with a controlled risk (R = 1.82%)
