"""Pricing and configuration constants for Claude Code usage analysis.

Pricing data is loaded from pricing.json for easy maintenance.
Subscription fees are stored in .fee.env (per-user, gitignored).
"""

import json
import sys
from pathlib import Path

# Load pricing data from JSON file
_PRICING_FILE = Path(__file__).parent / 'pricing.json'

# Subscription fee file (per-user, not version-controlled)
FEE_ENV_FILE = Path(__file__).parent / '.fee.env'

_FEE_KEYS = {
    'CLAUDE_MONTHLY_FEE': 'claude',
    'CODEX_MONTHLY_FEE': 'codex',
    'GEMINI_MONTHLY_FEE': 'gemini',
}


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


def load_subscription_fees():
    """Load subscription fees from .fee.env.

    Returns:
        dict or None: {'claude': float, 'codex': float, 'gemini': float},
                      or None if the file is missing or invalid.
    """
    if not FEE_ENV_FILE.exists():
        return None
    try:
        fees = {}
        with open(FEE_ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key in _FEE_KEYS:
                    fees[_FEE_KEYS[key]] = float(value)
        # Validate all three keys are present
        if set(fees.keys()) != {'claude', 'codex', 'gemini'}:
            return None
        return fees
    except (ValueError, OSError):
        return None


def save_subscription_fees(fees):
    """Write subscription fees to .fee.env.

    Args:
        fees: dict with keys 'claude', 'codex', 'gemini' mapping to floats.
    """
    with open(FEE_ENV_FILE, 'w') as f:
        f.write(f"CLAUDE_MONTHLY_FEE={fees['claude']}\n")
        f.write(f"CODEX_MONTHLY_FEE={fees['codex']}\n")
        f.write(f"GEMINI_MONTHLY_FEE={fees['gemini']}\n")


def prompt_subscription_fees():
    """Interactively prompt the user for monthly subscription fees.

    Returns:
        dict: {'claude': float, 'codex': float, 'gemini': float}
    """
    if not sys.stdin.isatty():
        print("Error: .fee.env not found and stdin is not a terminal.")
        print(f"Create {FEE_ENV_FILE} manually with:")
        print("  CLAUDE_MONTHLY_FEE=200")
        print("  CODEX_MONTHLY_FEE=200")
        print("  GEMINI_MONTHLY_FEE=19.99")
        sys.exit(1)

    print("Subscription fee configuration not found.")
    print("Please enter your monthly subscription fees:\n")

    prompts = [
        ('claude', 'Claude Code (Max)  monthly fee', '200'),
        ('codex',  'OpenAI Codex (Pro) monthly fee', '200'),
        ('gemini', 'Gemini CLI         monthly fee', '19.99'),
    ]

    fees = {}
    for key, label, default in prompts:
        while True:
            try:
                raw = input(f"  {label} [${default}]: ").strip()
                if not raw:
                    raw = default
                fees[key] = float(raw)
                break
            except ValueError:
                print(f"    Invalid number, please try again.")

    save_subscription_fees(fees)
    print(f"\nSaved to {FEE_ENV_FILE}")
    print("(Make sure .fee.env is in your .gitignore)\n")
    return fees


# Load all pricing data
_pricing_data = _load_pricing()

# ============================================================================
# CLAUDE (Anthropic) PRICING
# ============================================================================
MODEL_PRICING = _extract_model_pricing(_pricing_data['claude']['models'])
DEFAULT_PRICING = _extract_default_pricing(_pricing_data['claude']['default'])

# ============================================================================
# CODEX (OpenAI) PRICING
# ============================================================================
CODEX_MODEL_PRICING = _extract_model_pricing(_pricing_data['codex']['models'])
CODEX_DEFAULT_PRICING = _extract_default_pricing(_pricing_data['codex']['default'])

# ============================================================================
# GEMINI (Google) PRICING
# ============================================================================
GEMINI_MODEL_PRICING = _extract_model_pricing(_pricing_data['gemini']['models'])
GEMINI_DEFAULT_PRICING = _extract_default_pricing(_pricing_data['gemini']['default'])
