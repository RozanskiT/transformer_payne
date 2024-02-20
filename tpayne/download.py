def download_hf_model(model_name: str = None):
    """Download a model hosted on HuggingFace

    Args:
        model_name (str): HuggingFace model name

    Raises:
        ImportError: if huggingface-hub or joblib is not installed
    """
    try:
        from huggingface_hub import hf_hub_download
    except:
        raise ImportError(
            "Please install the `huggingface_hub` package to download HuggingFace models"
        )
        
    try:
        import joblib
    except:
        raise ImportError(
            "Please install the `joblib` package to download HuggingFace models"
        )

    if model_name is None:
        REPO_ID = "RozanskiT/transformer_payne"
        FILENAME = "TransformerPayneIntensities_v1.pkl"

    model = joblib.load(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    )
    
    return model
