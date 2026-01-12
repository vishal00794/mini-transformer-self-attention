# mini-transformer-self-attention
# Attention Is All You Need (Even on CPU)

This project demonstrates **why attention—not scale—is the key mechanism behind modern Large Language Models (LLMs)**. Instead of training a huge transformer, we build **small, CPU-trainable models** and isolate the *exact contribution* of attention.

We compare three architectures trained under **identical conditions**:

1. **No Attention** (BiLSTM baseline)
2. **Single-Head Attention** (BiLSTM + additive attention)
3. **Multi-Head Self-Attention (QKV)** (BiLSTM + Transformer-style attention)

The only difference between models is the **attention mechanism**. Same data, same parameters, same training budget.

---

##  Goal

To answer one core question:

> *Why does attention make language models dramatically better—even when models are small and trained on CPU?*



---

##  Dataset

* **IMDb Movie Reviews (25k samples)**
* Binary sentiment classification (positive / negative)
* Real English text (not toy data)

Preprocessing:

* Tokenization
* Top-10k vocabulary
* Padding/truncation to fixed length

---

##  Model Architectures

### 1️⃣ Baseline: BiLSTM (No Attention)

**Pipeline**:

```
Text → Embedding → BiLSTM → Max Pool → Classifier
```

**Characteristics**:

* Reads text left→right and right→left
* Produces contextual word representations
* Treats all words equally during pooling

**Limitation**:

* Cannot explicitly focus on important words
* Long sequences dilute key signals

---

### 2️⃣ Single-Head Attention (Additive Attention)

**Pipeline**:

```
Text → Embedding → BiLSTM → Attention → Context Vector → Classifier
```

**What changes**:

* Each word receives a learned *importance score*
* Softmax converts scores into attention weights
* Sentence representation becomes a **weighted sum** of words

**What it learns**:

* Which words matter most *globally*
* Suppresses noise words ("the", "is", etc.)

**Limitation**:

* One global notion of importance
* No word-to-word interaction

---

### 3️⃣ Multi-Head Self-Attention (QKV)

**Pipeline**:

```
Text → Embedding → BiLSTM → Multi-Head Self-Attention → Pooling → Classifier
```

**Key upgrade**:

* Uses **Query (Q), Key (K), Value (V)** projections
* Each word attends to every other word
* Multiple heads learn different linguistic relations

**Internally**:

```
Q = H · W_Q
K = H · W_K
V = H · W_V
Attention = softmax(QKᵀ / √d) · V
```

**Why this matters**:

* Captures negation ("not good")
* Models long-range dependencies
* Parallel attention heads learn syntax, sentiment, emphasis

This is **true Transformer-style attention**.

---

##  What We Compare

All models are trained with:

* Same dataset
* Same vocabulary
* Same embedding size
* Same optimizer & learning rate
* Same training steps

We compare:

* **Validation loss curves**
* **Test accuracy, F1, precision, recall, ROC-AUC**
* **Generated predictions vs ground truth**
* **Attention heatmaps (interpretability)**

---

##  Key Results (Qualitative)

### No Attention

* Fluent but generic predictions
* Misses key sentiment cues
* Sensitive to sentence length

### Single Attention

* Clearly focuses on sentiment words
* Better stability and accuracy
* Still limited by global attention

### Multi-Head Attention (QKV)

* Best overall performance
* Understands negation and structure
* Produces most coherent and accurate outputs

**Conclusion**:

> Even small models show that attention—not scale—is the real breakthrough.

---

##  Why This Project Matters

* Proves core LLM ideas without GPUs
* Shows *mechanism*, not just metrics
* Bridges RNNs → Transformers


---

##  How to Run

1. Open in **Google Colab (CPU is enough)**
2. Install dependencies (PyTorch)
3. Run notebook top-to-bottom
4. Observe loss curves, metrics, and attention heatmaps

---

##  Takeaway

> You don’t need billions of parameters to understand why LLMs work.
> Attention alone—properly isolated—explains most of the magic.

---

⭐ If you find this useful, consider starring the repo.

