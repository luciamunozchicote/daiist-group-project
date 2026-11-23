"""
The € formula (business case): when does cleaning a soiled panel pay off?

Clean when the energy the dirt costs, until the next scheduled cleaning, is worth
more than cleaning:

    clean if  loss x energy per m2 until next cleaning x price  >  cleaning cost per m2
    =>  loss threshold = cleaning cost / (energy per m2 until next cleaning x price)

Used by the app (pipeline.decide) and to evaluate component 2
(tools/train_component2.py), so both use the same threshold.
"""

CLEANING_EUR_PER_M2 = 0.09        # Micheli et al., Energy 2021 (1 MW plant, Granada), doi:10.1016/j.energy.2020.119018
YIELD_KWH_PER_KWP_YEAR = 1718     # same plant, 2019 AC yield
MODULE_KWP_PER_M2 = 0.21          # ~21% efficient module (team decision, 2026-11-23)
DAYS_TO_NEXT_CLEANING = 30        # team decision (2026-11-23); sensitivity 15/60/90 days in REPORT.md
PRICE_EUR_PER_KWH = 0.0325        # Spain solar PPA, LevelTen Q4 2025 (main case)
PRICE_SENSITIVITY = {             # OMIE solar captured price, UNEF annual report
    "OMIE solar captured 2025": 0.03639,
    "OMIE solar captured H1 2026": 0.02968,
}


def energy_value_per_m2(price: float = PRICE_EUR_PER_KWH, days: int = DAYS_TO_NEXT_CLEANING) -> float:
    """€ that one clean m2 of panel produces until the next scheduled cleaning."""
    return YIELD_KWH_PER_KWP_YEAR * MODULE_KWP_PER_M2 / 365 * days * price


def loss_threshold(price: float = PRICE_EUR_PER_KWH, days: int = DAYS_TO_NEXT_CLEANING) -> float:
    """Share of power lost (0-1) above which cleaning pays off."""
    return CLEANING_EUR_PER_M2 / energy_value_per_m2(price, days)


def net_gain_per_m2(loss: float, price: float = PRICE_EUR_PER_KWH, days: int = DAYS_TO_NEXT_CLEANING) -> float:
    """€ per m2 gained by cleaning now a panel that loses `loss` (0-1). Negative: not worth it."""
    return loss * energy_value_per_m2(price, days) - CLEANING_EUR_PER_M2


CLEAN_IF_LOSS_ABOVE = loss_threshold()
