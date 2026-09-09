"""zai-adapter: OpenAI-compatible facade over the signed ZCode protocol."""
from .signing import Signer, solve_pow
from .translate import openai_to_anthropic, anthropic_to_openai_response

__version__ = "1.0.0"
__all__ = ["Signer", "solve_pow", "openai_to_anthropic", "anthropic_to_openai_response", "__version__"]
