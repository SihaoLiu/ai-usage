"""Pricing and configuration constants for Claude Code usage analysis.

Pricing data is loaded from pricing.json for easy maintenance.
Update pricing.json when API rates or subscription prices change.
"""

import json
from pathlib import Path

# Load pricing data from JSON file
_PRICING_FILE = Path(__file__).parent / 'pricing.json'

def _load_pricing():
    """Load pricing configuration from JSON file."""
    with open(_PRICING_FILE, 'r') as f:
        return json.load(f)

def _extract_model_pricing(models_dict):
    """Extract model pricing, removing any _comment fields."""
    return {
        model: {k: v for k, v in pricing.items() if not k.startswith('_')}
        for model, pricing in models_dict.items()
    }

def _extract_default_pricing(default_dict):
    """Extract default pricing, removing any _comment fields."""
    return {k: v for k, v in default_dict.items() if not k.startswith('_')}

# Load all pricing data
_pricing_data = _load_pricing()

# ============================================================================
# CLAUDE (Anthropic) PRICING
# ============================================================================
MODEL_PRICING = _extract_model_pricing(_pricing_data['claude']['models'])
DEFAULT_PRICING = _extract_default_pricing(_pricing_data['claude']['default'])
SUBSCRIPTION_PRICE = _pricing_data['claude']['subscription_price']

# ============================================================================
# CODEX (OpenAI) PRICING
# ============================================================================
CODEX_MODEL_PRICING = _extract_model_pricing(_pricing_data['codex']['models'])
CODEX_DEFAULT_PRICING = _extract_default_pricing(_pricing_data['codex']['default'])
CODEX_SUBSCRIPTION_PRICE = _pricing_data['codex']['subscription_price']

# ============================================================================
# GEMINI (Google) PRICING
# ============================================================================
GEMINI_MODEL_PRICING = _extract_model_pricing(_pricing_data['gemini']['models'])
GEMINI_DEFAULT_PRICING = _extract_default_pricing(_pricing_data['gemini']['default'])
GEMINI_SUBSCRIPTION_PRICE = _pricing_data['gemini']['subscription_price']
