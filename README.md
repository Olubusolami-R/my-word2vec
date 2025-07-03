# my-word2vec Project Progress: Word2Vec from Scratch (Skip-gram + Negative Sampling)**

#### **1. Dataset Preparation**

* Used a **movie plot dataset** (combining all plots).
* **Preprocessing Steps:**

  * Lowercased all text.
  * Removed punctuation using `.isalpha()`.
  * Tokenised by splitting words.
* **Problem:** Unsure if sentence boundaries mattered.

  * **Solved:** Found that in Word2Vec, **order of words matters**, but not sentence boundaries.

---

#### **2. Vocabulary Building**

* Built a vocabulary dictionary (`word2idx`) by assigning a unique index to each word.
* Created `idx2word` for reverse lookup.
* Counted word frequencies.
* **Problem:** Model threw an **index out of range** error during training.

  * **Root Cause:** Indexing started from 1 instead of 0, making indices exceed vocab size.
  * **Solved:** Fixed by starting indexing from 0 (`count_idx = 0`).

---

#### **3. Training Pair Generation (Skip-gram)**

* Generated `(centre, context)` pairs using a sliding window approach.
* **Window Size:** 2.
* **Problem:** Very large number of training pairs → long processing time.

  * **Solution:** Started with a smaller dataset for debugging.

---

#### **4. Negative Sampling**

* Implemented `prepare_batch_with_negatives()`:

  * For every positive pair, added 5 negative samples (`k = 5`).
* **Problem:** Index out of range again during training.

  * **Solved:** Added `assert` checks for indices.
  * Root Cause: Indexing issue traced back to incorrect vocab indexing (fixed already).

---

#### **5. Model Building (PyTorch)**

* Defined `Word2Vec` class with:

  * Input embedding matrix.
  * Output embedding matrix.
* Forward pass:

  * Looked up embeddings for centre & context.
  * Computed **dot product** (summed over embedding dimensions).
* **Problem:** Unsure why the dot product involves summing.

  * **Solved:** Learned that sum along `dim=1` computes dot product across embedding dimensions.

---

#### **6. Training Loop**

* Trained using:

  * **BCELoss**.
  * **SGD optimiser** (learning rate `0.01`).
  * Batch size: 100.
  * 5 → then 25 epochs.
* **Problem:** Loss fluctuated and was slow to reduce.

  * **Solution:** Found that loss decrease in Word2Vec can be slow; what matters more is embedding quality, not just loss.

---

#### **7. Debugging Loss Tracking**

* Fixed tracking of **average loss per epoch**:

  * Reset `total_loss` at the start of each epoch.
  * Averaged by dividing by the number of batches.
* **Problem:** Mistakenly didn't reset total loss per epoch at first.

  * **Solved:** Fixed by resetting at the start of each epoch.

---

#### **8. Started Hyperparameter Tuning**

* Experimented with:
  
  * Number of negative samples (k)
  * Learning rate
  * Number of training epochs
* Kept track of loss values to monitor training behaviour.

---
