# Emotion Recognition from Speech

This project classifies the emotion expressed in a short spoken audio clip (angry, calm, disgust, fearful, happy, neutral, sad, surprised) using the RAVDESS dataset.

## Approach

- Extracted MFCC (Mel-Frequency Cepstral Coefficient) features from each audio clip — a compact numerical representation of pitch, tone, and texture.
- Used a **speaker-independent train/test split**: 5 actors were held out entirely for testing, while the remaining actors were used for training. This ensures the model is evaluated on voices it has never heard before.
- Scaled features using `StandardScaler`.
- Built a baseline **Logistic Regression** model.
- Built an **XGBoost** model for comparison.
- Evaluated using per-class precision, recall, F1-score, and a confusion matrix — not just overall accuracy, since some emotions are inherently harder to distinguish than others.

## Why a speaker-independent split matters

If the same actor's voice appears in both training and testing, the model can learn to recognize *whose voice it is* rather than *what emotion is being expressed* — inflating accuracy in a way that doesn't reflect real-world performance. Holding out entire actors (`Actor_15, Actor_13, Actor_11, Actor_07, Actor_02`) for testing ensures the reported results reflect genuine generalization to unseen speakers.

## Results

**Logistic Regression:** 38% accuracy
**XGBoost:** 38% accuracy

*(Note: with 8 balanced emotion classes, random guessing would score ~12.5%, so both models are learning real signal — but the accuracy is modest, which is expected and discussed below.)*

### Per-class performance (XGBoost)

| Emotion | Precision | Recall | F1-score |
|---|---|---|---|
| Angry | 0.49 | 0.75 | 0.59 |
| Calm | 0.47 | 0.40 | 0.43 |
| Disgust | 0.41 | 0.33 | 0.36 |
| Fearful | 0.69 | 0.28 | 0.39 |
| Happy | 0.40 | 0.30 | 0.34 |
| Neutral | 0.20 | 0.15 | 0.17 |
| Sad | 0.35 | 0.28 | 0.31 |
| Surprised | 0.22 | 0.45 | 0.30 |

**Angry** was the best-recognized emotion (highest recall, 0.75) — likely because it has the most distinct acoustic signature (louder, sharper pitch changes). **Neutral** performed worst — this is partly because it has half the training samples of other classes (RAVDESS doesn't include an intensity variant for neutral), and its acoustic profile is easily confused with "calm."

## What the model confuses

The confusion matrix showed frequent overlap between **surprised and angry/sad**, and between **calm and disgust** — emotions that share similar pitch and energy patterns when only average MFCC values are used (rather than how the sound changes over time). This suggests the model captures general vocal intensity more than nuanced emotional content.

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) — 1,440 audio clips across 24 actors and 8 emotions, downloaded via Kaggle.

## Model Card

**Intended use:** Educational/portfolio demonstration of an audio classification pipeline. Not validated for real-world deployment.

**Limitations:**
- Trained on RAVDESS, which uses **actors performing** emotions on cue — not spontaneous, real-life emotional speech. Performance would likely degrade significantly on natural conversation.
- Averaging MFCCs over each clip discards how the sound changes over time; a sequence model (LSTM) or CNN on spectrograms would likely capture emotional nuance better but requires substantially more data and training time than was available here.
- Accuracy is modest (38%) and should not be treated as production-grade; it is presented honestly as a baseline.
- **Must never be used** for real psychological assessment, hiring decisions, surveillance, lie detection, or any application inferring a real person's emotional or mental state. Emotion recognition from voice is scientifically contested and prone to bias.

## Technologies Used

- Python
- Librosa (audio feature extraction)
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn, Matplotlib

## Future Improvements

With more time, I would:
- Try a CNN on spectrograms or an LSTM on time-series MFCC sequences instead of averaged features.
- Test on additional datasets (TESS, EMO-DB) to check generalization beyond RAVDESS.
- Address class imbalance for the "neutral" category.
- Apply data augmentation (pitch shift, noise injection) to improve robustness.

## Project Notebook

The complete implementation is available in the repository notebook.
