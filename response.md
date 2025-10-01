1. What should you do if the two models have different tokenizers?

If the two models have different tokenizers, there are several approaches to handle this:

**Option 1: Vocabulary alignment**
- Create a mapping between the vocabularies of both tokenizers
- For tokens that exist in both vocabularies, use direct mapping
- For tokens that only exist in one vocabulary, either:
  - Map to a special "unknown" token
  - Use the closest semantic equivalent
  - Skip the contrastive computation for those tokens

**Option 2: Logit space alignment**
- Compute logits from both models using their respective tokenizers
- Project both logit distributions to a common vocabulary space
- This requires more complex alignment but preserves the semantic meaning better

2. Do you think contrastive decoding is used in practice?

No because the following reasons

- **Computational cost**: Requires running two models simultaneously, roughly doubling inference cost
- **Memory requirements**: Both models need to be loaded in memory
- **Latency**: Generation is slower due to dual model inference
- **Hyperparameter sensitivity**: Requires careful tuning of alpha to correct for both false positive and false negative failures.

