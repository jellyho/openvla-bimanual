def predict_action_chunk(vla, input_ids=None, unnorm_key=None, **kwargs):
    """Thin wrapper around super().generate() that decodes predicted actions and de-normalizes them."""

    # We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
    # in order for the predictions to match the training configuration and be accurate.
    input_ids = torch.cat(
        (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
    )

    # Run VLA inference
    ##$#$#$#$# changed inference length
    generated_ids = vla.generate(input_ids, max_new_tokens=vla.get_action_dim(unnorm_key) * self.Action_Length,
                                    min_new_tokens=vla.get_action_dim(unnorm_key) * self.Action_Length, **kwargs)
    print(generated_ids.shape)
    # Extract predicted action tokens and translate into (normalized) continuous actions
    predicted_action_token_ids = generated_ids[0, -vla.get_action_dim(unnorm_key) * self.Action_Length :].cpu().numpy()
    
    discretized_actions = vla.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized_actions = vla.bin_centers[discretized_actions]
    normalized_actions = normalized_actions.reshape(self.Action_Length, -1)     

    # Unnormalize actions
    unnormalized_actions = []
    for action in normalized_actions:
        action_norm_stats = vla.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (action + 1) * (action_high - action_low) + action_low,
            action,
        ) 
        unnormalized_actions.append(actions)       

    return np.vstack(unnormalized_actions)