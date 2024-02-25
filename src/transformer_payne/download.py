from transformer_payne.architecture_definition import ArchitectureDefinition

def download_hf_model(repository_id: str = None, filename: str = None) -> ArchitectureDefinition:
    """Download a model hosted on HuggingFace

    Args:
        repository_id (str): HuggingFace repository name "user/model_name"
        filename (str): the name of the weights/checkpoint file

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

    model_dict = joblib.load(
        hf_hub_download(repo_id=repository_id, filename=filename)
    )
    
    return ArchitectureDefinition.from_dict_config(model_dict)
