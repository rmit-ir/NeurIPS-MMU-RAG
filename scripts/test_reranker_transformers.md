# Reranker Model - compute_logits Function Explanation

## Overview

The `compute_logits` function takes model inputs and computes relevance scores for document ranking. It uses a causal language model to predict whether a document is relevant to a query by looking at the model's prediction for the last token (which should be either "yes" or "no").

## Step-by-Step Dimension Analysis

### Step 1: Model Output

```python
batch_scores = model(**inputs).logits[:, -1, :]
```

**Input to model**: Tokenized text sequences

- Shape: `[batch_size, sequence_length]`

**Model output logits**: 

- Shape: `[batch_size, sequence_length, vocab_size]`
- Example: `[32, 512, 151936]` (32 documents, 512 tokens each, 151936 vocabulary size)

**After slicing `[:, -1, :]`** (last token position):

- Shape: `[batch_size, vocab_size]`
- Example: `[32, 151936]`
- This gives us the logits for predicting the next token for each document in the batch

### Step 2: Extract Yes/No Token Logits

```python
true_vector = batch_scores[:, token_true_id]    # "yes" token
false_vector = batch_scores[:, token_false_id]  # "no" token
```

**Each vector shape**: `[batch_size]`

- `true_vector`: Logits for "yes" token for each document
- `false_vector`: Logits for "no" token for each document

### Step 3: Stack into Binary Classification Format

```python
batch_scores = torch.stack([false_vector, true_vector], dim=1)
```

**After stacking**:

- Shape: `[batch_size, 2]`
- Example: `[32, 2]`
- Column 0: "no" logits, Column 1: "yes" logits

### Step 4: Apply Log Softmax

```python
batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
```

**After log_softmax**:

- Shape: `[batch_size, 2]`
- Values: Log probabilities (negative numbers)
- Each row sums to a log probability of 1

### Step 5: Extract "Yes" Probability

```python
scores = batch_scores[:, 1].exp().tolist()
```

**Final scores**:

- Shape: `[batch_size]`
- Values: Probabilities between 0 and 1
- Each score represents P(relevant=yes) for each document

## Visual Representation

```
Original Logits: [batch_size, vocab_size]
                 ↓
Extract "yes"/"no": [batch_size] each
                 ↓
Stack: [batch_size, 2]
       [ [P(no), P(yes)],
         [P(no), P(yes)],
         ... ]
                 ↓
LogSoftmax: [batch_size, 2] (log probabilities)
                 ↓
Exp: [batch_size] (final probabilities for "yes")
```

## Example with Concrete Numbers

For a batch of 3 documents:
```
Step 1 - Raw logits for last token: [3, 151936]
Step 2 - Extract yes/no logits:
  true_vector = [2.1, -0.5, 1.8]  (logits for "yes")
  false_vector = [-0.3, 1.2, -1.1] (logits for "no")
Step 3 - Stacked: 
  [[-0.3, 2.1],
   [1.2, -0.5],
   [-1.1, 1.8]]
Step 4 - LogSoftmax:
  [[-2.7, -0.2],  # P(no)=0.067, P(yes)=0.819
   [-2.0, -0.3],  # P(no)=0.135, P(yes)=0.741
   [-3.2, -0.1]]  # P(no)=0.041, P(yes)=0.905
Step 5 - Final scores (exp of yes column):
  [0.819, 0.741, 0.905]
```

This gives us relevance scores where higher values indicate more relevant documents.
