import transformers as tr
import torch
import torch.nn.functional as F
from typing import Optional

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def load_model(model_name):
    model = tr.AutoModelForCausalLM.from_pretrained(
        model_name,
		torch_dtype=torch.float16,
		device_map="auto",
		trust_remote_code=True
	)
    return model

def contrastive_generation(
    amateur_model, 
    expert_model, 
    prompt: str, 
    max_tokens: int,
    alpha: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> str:
    """
    Implements token-level contrastive decoding.
    
    Args:
        amateur_model: The smaller/weaker model
        expert_model: The larger/stronger model  
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        alpha: Adaptive plausibility constraint 
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
    
    Returns:
        Generated text string
    """
    device = next(expert_model.parameters()).device
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    generated_tokens = input_ids.clone()
    
    # Initialize past key values for caching
    amateur_past_key_values = None
    expert_past_key_values = None
    
    with torch.no_grad():
        for step in range(max_tokens):
            # For the first step, use the full sequence; for subsequent steps, use only the last token
            if step == 0:
                amateur_input_ids = generated_tokens
                expert_input_ids = generated_tokens
                amateur_attention_mask = attention_mask
                expert_attention_mask = attention_mask
            else:
                amateur_input_ids = generated_tokens[:, -1:]
                expert_input_ids = generated_tokens[:, -1:]
                # Update the full attention mask for both models
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)], dim=-1)
                amateur_attention_mask = attention_mask
                expert_attention_mask = attention_mask


            # Get logits from both models with caching
            amateur_outputs = amateur_model(
                input_ids=amateur_input_ids,
                attention_mask=amateur_attention_mask,
                past_key_values=amateur_past_key_values,
                use_cache=True
            )
            expert_outputs = expert_model(
                input_ids=expert_input_ids,
                attention_mask=expert_attention_mask,
                past_key_values=expert_past_key_values,
                use_cache=True
            )

            # Update past key values for next iteration
            amateur_past_key_values = amateur_outputs.past_key_values
            expert_past_key_values = expert_outputs.past_key_values

            # Get the logits for the last token
            amateur_logits = amateur_outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            expert_logits = expert_outputs.logits[:, -1, :]    # [batch_size, vocab_size]
        


            expert_log_probs = F.log_softmax(expert_logits, dim=-1)
            amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)

            # Calculate max probability for the expert
            expert_probs = F.softmax(expert_logits, dim=-1) # [batch_size, vocab_size]
            max_expert_prob = torch.max(expert_probs, dim=-1, keepdim=True).values # [batch_size, 1]

            # Create a mask for tokens that meet the plausibility constraint
            plausibility_mask = (expert_probs >= alpha * max_expert_prob) # [batch_size, vocab_size]



            # Calculate CD-Score (log P_EXP - log P_AMA_temp)
            # CD-score(x_i) = log P_EXP(x_i) - log P_AMA_tau(x_i)
            contrastive_logits = expert_log_probs - amateur_log_probs # [batch_size, vocab_size]

            print(f"contrastive_logits shape: {contrastive_logits.shape}")

            # Apply the Plausibility Mask
            # Tokens failing the constraint get a score of -inf
            # The paper: CD-score(x_i) = -inf if x_i not in V_head(x_<i)
            contrastive_logits[~plausibility_mask] = -float('inf')



            # Apply temperature
            if temperature != 1.0:
                contrastive_logits = contrastive_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(contrastive_logits, top_k, dim=-1)
                # Set all other logits to -inf
                filtered_logits = torch.full_like(contrastive_logits, float('-inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
                contrastive_logits = filtered_logits

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(contrastive_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Create a mask of tokens to keep based on cumulative probability
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Set logits to -inf for tokens to remove using the original indices
                contrastive_logits[sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)] = float('-inf')


            # Sample from the contrastive distribution or use greedy decoding
            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(contrastive_logits, dim=-1, keepdim=True)
            else:
                # Sampling
                probs = F.softmax(contrastive_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append the new token
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
    
    # Decode the generated tokens (excluding the original prompt)
    generated_text = tokenizer.decode(
        generated_tokens[0, input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text


def main(amateur_path, expert_path):
    amateur_model = load_model(amateur_path)
    expert_model = load_model(expert_path)
    
