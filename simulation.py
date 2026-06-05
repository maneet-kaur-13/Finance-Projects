import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PortfolioConfig:
    n_policies: int = 1000
    years: int = 1
    expected_claim_frequency_per_policy: float = 0.08
    avg_claim_severity: float = 50000.0
    severity_sigma: float = 0.9
    inflation_factor: float = 1.0
    deductible: float = 0.0
    policy_limit: float = 200000.0
    premium_per_policy: float = 7000.0
    fixed_expenses: float = 1200000.0


@dataclass
class StressScenario:
    frequency_multiplier: float = 1.0
    severity_multiplier: float = 1.0
    name: str = "Base"


class InsuranceRiskSimulator:
    def __init__(self, config: PortfolioConfig, random_seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(random_seed)

    def simulate_one_iteration(self, scenario: StressScenario) -> Tuple[int, float, float]:
        cfg = self.config

        # Total number of claims in the portfolio over the horizon.
        portfolio_lambda = (
            cfg.n_policies
            * cfg.expected_claim_frequency_per_policy
            * cfg.years
            * scenario.frequency_multiplier
        )
        n_claims = self.rng.poisson(portfolio_lambda)

        if n_claims == 0:
            gross_loss = 0.0
            net_underwriting_result = cfg.n_policies * cfg.premium_per_policy - cfg.fixed_expenses
            return 0, gross_loss, net_underwriting_result

        # Lognormal gives a realistic right-skewed claim severity distribution.
        severities = self.rng.lognormal(
            mean=np.log(cfg.avg_claim_severity * scenario.severity_multiplier * cfg.inflation_factor) - 0.5 * cfg.severity_sigma**2,
            sigma=cfg.severity_sigma,
            size=n_claims,
        )

        # Apply deductible and policy limit.
        severities = np.maximum(severities - cfg.deductible, 0)
        severities = np.minimum(severities, cfg.policy_limit)

        gross_loss = float(np.sum(severities))
        premium_income = cfg.n_policies * cfg.premium_per_policy
        net_underwriting_result = premium_income - gross_loss - cfg.fixed_expenses
        return n_claims, gross_loss, net_underwriting_result

    def run_simulation(self, n_simulations: int, scenario: StressScenario) -> Dict[str, np.ndarray]:
        claim_counts = np.zeros(n_simulations)
        losses = np.zeros(n_simulations)
        underwriting_results = np.zeros(n_simulations)

        for i in range(n_simulations):
            n_claims, loss, uw_result = self.simulate_one_iteration(scenario)
            claim_counts[i] = n_claims
            losses[i] = loss
            underwriting_results[i] = uw_result

        return {
            "claim_counts": claim_counts,
            "losses": losses,
            "underwriting_results": underwriting_results,
        }


def var_cvar(losses: np.ndarray, confidence_level: float = 0.99) -> Tuple[float, float]:
    sorted_losses = np.sort(losses)
    var = float(np.quantile(sorted_losses, confidence_level))
    tail_losses = sorted_losses[sorted_losses >= var]
    cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var
    return var, cvar


def probability_of_ruin(underwriting_results: np.ndarray, capital_buffer: float) -> float:
    # Ruin occurs if underwriting loss exceeds the capital buffer.
    return float(np.mean(underwriting_results < -capital_buffer))


def summarize_results(results: Dict[str, np.ndarray], capital_buffer: float) -> Dict[str, float]:
    losses = results["losses"]
    uw = results["underwriting_results"]
    var_95, cvar_95 = var_cvar(losses, 0.95)
    var_99, cvar_99 = var_cvar(losses, 0.99)

    return {
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
        "p95_var": var_95,
        "p95_cvar": cvar_95,
        "p99_var": var_99,
        "p99_cvar": cvar_99,
        "mean_underwriting_result": float(np.mean(uw)),
        "probability_of_underwriting_loss": float(np.mean(uw < 0)),
        "probability_of_ruin": probability_of_ruin(uw, capital_buffer),
        "mean_claim_count": float(np.mean(results["claim_counts"])),
    }


if __name__ == "__main__":
    config = PortfolioConfig()
    simulator = InsuranceRiskSimulator(config=config, random_seed=42)
    capital_buffer = 2500000.0

    scenarios = [
        StressScenario(1.0, 1.0, "Base"),
        StressScenario(1.25, 1.0, "Higher Frequency"),
        StressScenario(1.0, 1.20, "Higher Severity"),
        StressScenario(1.25, 1.20, "Combined Stress"),
    ]

    for scenario in scenarios:
        results = simulator.run_simulation(n_simulations=10000, scenario=scenario)
        summary = summarize_results(results, capital_buffer=capital_buffer)
        print(f"\nScenario: {scenario.name}")
        for k, v in summary.items():
            print(f"{k}: {v:,.2f}")
