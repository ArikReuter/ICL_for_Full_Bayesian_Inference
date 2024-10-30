def make_basic_model_config(P:int, use_intercept:bool = False) -> dict:
    """
    make a basic model configuration for the CFM model.
    Args:
    P: int: number of features per in-context dataset
    use_intercept: bool: whether to use an intercept in the model
    Returns:
    dict: the model configuration
    """
    P1 = P + 1 if use_intercept else P
    config = {
            "Type": "TransformerCNFConditionalDecoder", 
            "n_input_features_encoder": P+1,
            "n_input_features_decoder": P1,
            "d_model_encoder": 512,
            "d_model_decoder": 512,
            "n_heads_encoder": 8,
            "n_heads_decoder": 8,
            "d_ff_encoder": 1024,
            "d_ff_decoder": 1024,
            "dropout_encoder": 0.1,
            "dropout_decoder": 0.1,
            "n_conditional_input_features":  1,
            "n_condition_features": 512,
            "n_layers_condition_embedding": 3,
            "n_layers_encoder": 8,
            "n_layers_decoder": 6,
            "use_positional_encoding_encoder": True,
            "use_positional_encoding_decoder": False,
            "use_self_attention_decoder": False,
            "output_dim": P1,
            "d_final_processing": 512,
            "n_final_layers": 3,
            "dropout_final": 0.1,
            "treat_z_as_sequence": False,
        }
    
    return config