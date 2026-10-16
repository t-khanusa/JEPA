import hydra
from omegaconf import OmegaConf

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

def replace_substring(value: str, old: str, new: str) -> str:
    return str(value).replace(str(old), str(new))

def span_tag(value) -> str:
    """'' when training.lyap_frames is unset, else 'T<n>' (keeps old run names unchanged)."""
    return "" if value in (None, "None", "null") else f"T{value}"

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)
OmegaConf.register_new_resolver("replace_substring", replace_substring)
OmegaConf.register_new_resolver("span_tag", span_tag)

if __name__ == "__main__":
    register_resolvers()

