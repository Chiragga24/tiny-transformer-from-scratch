from __future__ import annotations

import torch

from tiny_transformer.model import TransformerLM


@torch.no_grad()
def generate_token_ids(
    model: TransformerLM,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
    precision: str = "fp32",
) -> list[int]:
    """Autoregressively extend a prompt and return prompt plus generated IDs.

    Temperature controls randomness. A value of zero performs greedy decoding;
    otherwise top-k sampling restricts each choice to the k most likely tokens.
    """
    if not prompt_token_ids:
        raise ValueError("the prompt must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")

    device = next(model.parameters()).device
    generated = torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
    model.eval()

    for _ in range(max_new_tokens):
        # The model was trained only up to context_length positions. Once the
        # sequence grows beyond that window, retain the most recent tokens.
        model_input = generated[-model.context_length :].unsqueeze(0)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            next_token_logits = model(model_input)[0, -1]

        if temperature == 0:
            next_token = torch.argmax(next_token_logits).reshape(1)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                effective_k = min(top_k, next_token_logits.numel())
                threshold = torch.topk(next_token_logits, effective_k).values[-1]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator)

        generated = torch.cat((generated, next_token))
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated.tolist()
