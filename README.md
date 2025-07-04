# Word2Vec from Scratch (Skip-gram + Negative Sampling)

## Why This Project

I wanted to really understand how **Word2Vec** works under the hood, so I decided to implement it from scratch using PyTorch. Instead of using standard datasets like Text8, I went with a movie plot dataset because it felt more interesting and unpredictable. The goal here wasn’t just to make it work but to understand every part of the process.

**Note:** I did all the training on Kaggle using the free GPUs and also got the movie plot dataset from there.

---

## What I Did and Learned

### 1. Dataset Preparation

I combined all the movie plots into one large text. For preprocessing, I:

* Lowercased everything.
* Removed punctuation by keeping only words made up of letters (`.isalpha()`).
* Tokenised by splitting on spaces.

At first, I wasn’t sure whether sentence boundaries mattered for Word2Vec. Turns out, they don’t—what really matters is the order of words.


### 2. Building the Vocabulary

I created a mapping from words to indices (`word2idx`) and another for reverse lookup (`idx2word`). I also counted word frequencies.

I ran into an index out-of-range error during training. After some digging, I realised it was because I started indexing from 1 instead of 0. Fixing that solved the problem.


### 3. Generating Training Pairs (Skip-gram)

I used a sliding window approach (window size = 2) to generate centre-context word pairs.

One problem here was that the number of pairs got really large, and it slowed things down a lot. To manage that, I started with a smaller subset of the dataset to debug and test things before scaling up.


### 4. Negative Sampling

I wrote a function to prepare batches with negative samples. For each positive pair, I added 5 negative samples.

I hit another index error during training. After adding some `assert` checks, I traced it back to my earlier indexing mistake. Once that was fixed, this part worked fine.


### 5. Model Building (PyTorch)

I built a simple Word2Vec model with:

* An input embedding matrix.
* An output embedding matrix.

The forward pass:

* Looks up the embeddings for centre and context words.
* Computes the dot product between them by summing over the embedding dimensions.

At first, I wasn’t sure why summing was needed, but I learned that this is how the dot product works across embeddings.


### 6. Training the Model

I trained the model using:

* Binary cross-entropy loss (BCELoss).
* SGD optimiser with a learning rate of 0.01.
* Batch size of 100.
* Initially, 5 epochs, then I increased it to 25 and more.

The loss didn’t always go down smoothly. I learned that in Word2Vec, what matters more is the quality of the embeddings rather than just how low the loss gets.


### 7. Tracking Loss Properly

I also fixed how I tracked loss. Initially, I wasn’t resetting the total loss at the start of each epoch, which made the numbers confusing. Once I fixed that, I could properly see the average loss per epoch.


### 8. Hyperparameter Experiments

Lately, I’ve been experimenting with:

* Different numbers of negative samples (`k`).
* Learning rates.
* Different numbers of epochs.

I’ve been tracking how these changes affect the loss and training behaviour.

---

## Why I’m Sharing This

This project has been a hands-on way for me to learn not just Word2Vec, but also debugging, model building, and training workflows in PyTorch. I’m sharing it here to document my process, and maybe it’ll be useful for others who are also trying to learn by building things from scratch.

