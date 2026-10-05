# PersonaWear

## Personalized Multimodal Foundation Models for Robust and Explainable Wearable Health Monitoring

PersonaWear is a research-oriented AI system for personalized health monitoring using longitudinal wearable sensor data.

The project investigates whether multimodal wearable models can provide reliable health-state predictions across different users while remaining robust to missing sensors and providing calibrated, interpretable predictions.

PersonaWear focuses on four health dimensions:

- Stress
- Fatigue
- Sleep and recovery
- General wellness

Rather than relying only on population-level predictions, PersonaWear learns an individual's physiological baseline and adapts predictions to that person's historical patterns.

---

# Research Question

Can personalized multimodal wearable models improve health-state prediction while remaining robust to missing sensor modalities and providing well-calibrated, interpretable outputs?

The project investigates four major challenges:

1. Personalization across users
2. Multimodal sensor fusion
3. Missing or noisy wearable signals
4. Prediction uncertainty and explainability

---

# Core Idea

Traditional wearable ML systems typically learn:

Wearable measurements → Health label

PersonaWear instead uses:

Wearable measurements  
↓  
Multimodal temporal encoders  
↓  
Sensor-quality estimation  
↓  
Dynamic modality fusion  
↓  
Personalized user representation  
↓  
Health prediction  
↓  
Uncertainty estimation  
↓  
Explainable health evidence

---

# Supported Wearable Signals

PersonaWear can work with combinations of:

| Signal | Description |
|---|---|
| Heart Rate | Beats per minute |
| HRV | Heart-rate variability |
| PPG/BVP | Photoplethysmography |
| EDA | Electrodermal activity |
| Skin Temperature | Peripheral temperature |
| Accelerometer | Physical motion |
| SpO₂ | Blood-oxygen saturation |
| Sleep Duration | Total sleep |
| Sleep Stages | Awake / light / deep / REM |
| Steps | Daily activity |
| Activity Intensity | Sedentary/moderate/vigorous |
| Resting HR | Daily cardiovascular baseline |

The architecture should not require every sensor to be present.

---

# Example Input

```json
{
  "user_id": "subject_014",
  "timestamp": "2026-10-06T08:30:00",
  "heart_rate": 91,
  "hrv": 34,
  "skin_temperature": 36.4,
  "sleep_hours": 5.9,
  "steps": 1250,
  "activity": "low",
  "eda": null
}
```

Notice that EDA is unavailable.

PersonaWear should still produce a prediction instead of failing.

---

# Example Output

```json
{
  "stress": {
    "prediction": "high",
    "probability": 0.83
  },
  "fatigue": {
    "prediction": "moderate",
    "probability": 0.71
  },
  "recovery_score": 42,
  "confidence": 0.87,
  "available_modalities": [
    "heart_rate",
    "hrv",
    "temperature",
    "sleep",
    "activity"
  ],
  "missing_modalities": [
    "eda"
  ]
}
```

---

# Personalized Health Baseline

Population averages are often insufficient because physiological measurements vary significantly between individuals.

PersonaWear therefore learns a personal baseline.

For example:

```text
User's normal resting HR: 63 BPM
Current resting HR:       76 BPM

Difference: +20.6%
```

Instead of interpreting 76 BPM purely against a general population threshold, PersonaWear considers how unusual it is for that particular user.

Possible personalized features include:

```text
HR deviation from 14-day baseline

HRV deviation from baseline

sleep deviation

activity deviation

temperature deviation

recovery trend

rolling physiological statistics
```

---

# System Architecture

```text
                  Wearable Sensor Streams

 HR / HRV       PPG       EDA       TEMP       ACC       Sleep
    │            │         │          │          │          │
    ▼            ▼         ▼          ▼          ▼          ▼

              Temporal Sensor Encoders

                       │
                       ▼

               Sensor Quality Layer

                       │
                       ▼

              Modality Availability Mask

                       │
                       ▼

             Dynamic Multimodal Fusion

                       │
                       ▼

              Shared Health Embedding

                       │
               ┌───────┴────────┐
               │                │
               ▼                ▼

       Personalized Adapter   Global Model

               │
               └───────┬────────┘
                       ▼

                Health Predictor

             ┌─────────┼─────────┐
             ▼         ▼         ▼

           Stress    Fatigue   Recovery

                       │
                       ▼

               Uncertainty Module

                       │
                       ▼

               Explanation Module

                       │
                       ▼

              Health Insight Output
```

---

# Model Architecture

## 1. Sensor Encoders

Each sensor modality receives its own temporal encoder.

Possible encoders:

- 1D CNN
- LSTM
- GRU
- Temporal Convolution Network
- Transformer
- Time-series foundation model

Example:

```text
HR ───────── Transformer Encoder ──────┐

HRV ─────── Transformer Encoder ──────┤

EDA ─────── Transformer Encoder ──────┤

ACC ─────── Transformer Encoder ──────┤
                                       ├─ Fusion
Temperature ─ Transformer Encoder ────┤

Sleep ───── Transformer Encoder ──────┘
```

---

# Dynamic Sensor Fusion

One of PersonaWear's main research components is handling unavailable sensor modalities.

For every sample, construct a modality mask:

```text
HR      = 1
HRV     = 1
EDA     = 0
TEMP    = 1
ACC     = 1
Sleep   = 1
```

The fusion network learns which available signals should receive the highest attention.

Conceptually:

```text
Sensor representations
        +
Sensor availability mask
        +
Sensor quality score
        ↓
Dynamic Modality Router
        ↓
Weighted multimodal representation
```

Possible implementation approaches:

- Gated multimodal fusion
- Attention-based fusion
- Mixture-of-experts
- Modality dropout
- Cross-modal Transformer
- Learned sensor routing

---

# Personalization Layer

Three approaches should be compared.

### Global model

One model is shared across every participant.

```text
Everyone
   ↓
Shared Model
```

### Fine-tuned personalized model

```text
Global Model
    ↓
User-specific fine tuning
```

### Parameter-efficient personalization

Recommended research approach:

```text
Shared Foundation Model
        ↓
User Adapter / LoRA
        ↓
Personalized Prediction
```

This allows most model parameters to remain shared while learning small user-specific components.

---

# Prediction Tasks

PersonaWear can use a multi-task learning architecture.

```text
Shared Wearable Representation
              │
    ┌─────────┼──────────┬──────────┐
    ▼         ▼          ▼          ▼

 Stress    Fatigue    Recovery    Wellness
```

Example labels:

### Stress

```text
0 = Low
1 = Moderate
2 = High
```

### Fatigue

```text
0 = Normal
1 = Moderate
2 = Severe
```

### Recovery

Regression score:

```text
0–100
```

### Wellness

Regression or classification based on available questionnaire labels.

---

# Uncertainty Estimation

PersonaWear should not produce highly confident predictions when sensor evidence is weak.

Possible techniques:

### Monte Carlo Dropout

Run the model several times:

```text
Prediction 1 = 0.82
Prediction 2 = 0.79
Prediction 3 = 0.85
Prediction 4 = 0.64
Prediction 5 = 0.81
```

High variation indicates uncertainty.

### Deep Ensembles

Train several independently initialized models and compare their predictions.

### Temperature Scaling

Calibrate probabilities on a validation set.

Evaluation metrics:

- Expected Calibration Error
- Brier Score
- Negative Log-Likelihood
- Reliability diagrams

---

# Explainability

PersonaWear should explain what evidence contributed to a prediction.

Example:

```text
Prediction: High Stress

Confidence: 87%

Important evidence:

1. HRV decreased 24% from the user's baseline.
2. Resting HR increased 13%.
3. Sleep duration decreased by 1.7 hours.
4. Activity levels were below the user's usual level.
```

Possible techniques:

- SHAP
- attention visualization
- feature attribution
- Integrated Gradients
- temporal importance analysis

The explanations should be generated from model evidence rather than invented medical claims.

---

# Optional LLM Health Agent

The final layer can provide natural-language summaries.

```text
Sensor Data
      ↓
PersonaWear Prediction Model
      ↓
Structured Evidence
      ↓
LLM
      ↓
Readable Health Summary
```

Important:

The LLM should never receive only the raw question.

It should receive structured verified evidence.

Example:

```json
{
  "prediction": "high_stress",
  "confidence": 0.87,
  "evidence": {
    "hrv_change": "-24%",
    "resting_hr_change": "+13%",
    "sleep_change": "-1.7 hours"
  }
}
```

The LLM can then generate:

```text
Your physiological signals differ noticeably from your recent baseline.

HRV is lower than usual, resting heart rate is elevated, and your sleep duration was shorter than your recent average.

The model estimates an increased likelihood of physiological stress, with relatively high confidence.
```

PersonaWear should not diagnose medical conditions.

---

# Dataset Strategy

The project should initially use public datasets.

Possible categories:

```text
Dataset A
Stress + physiological signals

Dataset B
Sleep + wearable signals

Dataset C
Longitudinal health / activity

Dataset D
Large-scale wearable foundation-model data
```

Create a common representation:

```text
data/
├── raw/
│   ├── dataset_1/
│   ├── dataset_2/
│   └── dataset_3/
│
├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── metadata/
```

---

# Important Data Splitting Rule

Wearable datasets should be split by participant.

Bad approach:

```text
Random samples

User A → training
User A → testing
```

This can cause identity leakage.

Recommended approach:

```text
Train:
Users 1–70

Validation:
Users 71–85

Test:
Users 86–100
```

The test users should ideally never appear during training.

---

# Main Experiments

## Experiment 1 — Global vs Personalized

Compare:

```text
Global Model
vs
Fine-Tuned Model
vs
Adapter-Based Personalized Model
```

Measure:

- AUROC
- AUPRC
- F1
- accuracy
- calibration

---

# Experiment 2 — Sensor Ablation

Compare:

```text
HR

HR + HRV

HR + HRV + ACC

HR + HRV + ACC + Sleep

All sensors
```

Determine how much each modality contributes.

---

# Experiment 3 — Missing Sensors

Artificially remove sensor modalities.

Evaluate:

```text
0% missing

10% missing

25% missing

50% missing

Entire modality unavailable
```

Compare:

```text
Zero filling

Mean imputation

Masking

Modality dropout

Dynamic routing

PersonaWear
```

---

# Experiment 4 — Cross-User Generalization

Use Leave-One-Subject-Out evaluation.

```text
Train:
Everyone except User X

Test:
User X
```

Repeat across participants.

---

# Experiment 5 — Personalization Data Requirement

Determine how much user data is needed.

```text
0 minutes

5 minutes

15 minutes

30 minutes

1 hour

1 day

3 days

7 days
```

Measure how performance improves as personalization data increases.

This could become one of the most interesting findings in the paper.

---

# Experiment 6 — Calibration

Compare confidence with actual correctness.

Example:

```text
Predictions with 90% confidence

Expected:
~90% should actually be correct.
```

Compare calibration before and after personalization.

---

# Experiment 7 — Robustness to Sensor Noise

Artificially inject:

```text
Gaussian noise

sensor dropout

missing windows

motion artifacts

signal corruption
```

Test whether PersonaWear remains stable.

---

# Research Ablation Study

Recommended ablations:

| Model | Personalization | Dynamic Fusion | Uncertainty |
|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ |
| Model A | ✓ | ✗ | ✗ |
| Model B | ✗ | ✓ | ✗ |
| Model C | ✓ | ✓ | ✗ |
| PersonaWear | ✓ | ✓ | ✓ |

This lets the paper show exactly which components contribute to improvements.

---

# Evaluation Metrics

For classification:

```text
Accuracy
Macro F1
AUROC
AUPRC
Sensitivity
Specificity
```

For regression:

```text
MAE
RMSE
R²
Pearson correlation
```

For uncertainty:

```text
ECE
Brier Score
NLL
```

For missing sensors:

```text
Performance degradation percentage
```

---

# Suggested Technology Stack

## Machine Learning

```text
Python
PyTorch
PyTorch Lightning
scikit-learn
NumPy
Pandas
SciPy
```

## Time-Series Processing

```text
NeuroKit2
HeartPy
SciPy Signal
tsfresh
```

## Explainability

```text
SHAP
Captum
```

## Experiment Tracking

```text
Weights & Biases
MLflow
TensorBoard
```

## LLM Layer

Possible providers:

```text
Open-source LLM
Llama
Qwen
Mistral
```

The LLM component is optional and should remain separated from the physiological prediction model.

---

# Proposed Repository Structure

```text
PersonaWear/
│
├── README.md
│
├── configs/
│   ├── baseline.yaml
│   ├── multimodal.yaml
│   └── personalized.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── datasets/
│   ├── base_dataset.py
│   ├── stress_dataset.py
│   ├── sleep_dataset.py
│   └── multimodal_dataset.py
│
├── preprocessing/
│   ├── heart_rate.py
│   ├── hrv.py
│   ├── eda.py
│   ├── accelerometer.py
│   ├── sleep.py
│   └── normalize.py
│
├── models/
│   ├── encoders/
│   │   ├── temporal_cnn.py
│   │   ├── transformer.py
│   │   └── lstm.py
│   │
│   ├── fusion/
│   │   ├── early_fusion.py
│   │   ├── attention_fusion.py
│   │   └── dynamic_router.py
│   │
│   ├── personalization/
│   │   ├── user_embedding.py
│   │   └── adapter.py
│   │
│   ├── uncertainty/
│   │   ├── mc_dropout.py
│   │   └── calibration.py
│   │
│   └── personawear.py
│
├── explainability/
│   ├── shap_explainer.py
│   ├── attention_analysis.py
│   └── evidence_builder.py
│
├── agents/
│   ├── wearable_agent.py
│   └── prompts.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── losses.py
│
├── experiments/
│   ├── global_vs_personalized.py
│   ├── missing_sensor.py
│   ├── ablation.py
│   ├── calibration.py
│   └── cross_subject.py
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── signal_visualization.ipynb
│   └── results_analysis.ipynb
│
├── tests/
│
└── requirements.txt
```

---

# Minimum Viable Research Version

Do not begin by implementing everything.

Start with:

### Phase 1

```text
Dataset
↓
HR + HRV + ACC
↓
Transformer/LSTM
↓
Stress prediction
```

Establish the baseline.

### Phase 2

Add:

```text
Sleep
Temperature
EDA
```

Build multimodal fusion.

### Phase 3

Add:

```text
Personalized baseline
+
user adapter
```

### Phase 4

Add:

```text
sensor masking
+
modality dropout
+
dynamic sensor routing
```

### Phase 5

Add:

```text
uncertainty calibration
```

### Phase 6

Add:

```text
explainability
```

### Phase 7

Optionally add:

```text
LLM health agent
```

---

# Primary Hypotheses

## H1

Personalized models will outperform population-only models for wearable health prediction.

## H2

Dynamic multimodal fusion will outperform static fusion when one or more sensor modalities are unavailable.

## H3

Modality-dropout training will improve robustness to real-world sensor loss.

## H4

Personalization will improve prediction calibration in addition to classification performance.

## H5

Model-derived explanations will identify physiologically meaningful temporal and sensor-level evidence.

---

# Potential Paper Title


**PersonaWear: Robust Personalized Health Prediction from Incomplete Multimodal Wearable Signals**

The second title may be stronger academically because the research problem is immediately clear.

---

# Expected Research Contribution

PersonaWear aims to demonstrate that reliable wearable health monitoring requires more than simply adding additional sensors.

The project studies how personalization, multimodal representation learning, adaptive modality routing, and uncertainty estimation interact under realistic wearable-data conditions.

The principal contribution is a wearable-learning framework that can adapt to individual physiological baselines while maintaining performance when one or more sensor modalities are missing or unreliable.

---

# Disclaimer

PersonaWear is intended for research purposes.

It is not a medical diagnostic system and should not be used to replace professional medical advice, diagnosis, or treatment.
