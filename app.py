import argparse
import json
from simulation import PortfolioConfig, InsuranceRiskSimulator, StressScenario, summarize_results


def main():
    parser = argparse.ArgumentParser(description="Insurance Risk Simulation and Solvency Analysis")
    parser.add_argument("--n_policies", type=int, default=1000)
    parser.add_argument("--claim_freq", type=float, default=0.08)
    parser.add_argument("--avg_severity", type=float, default=50000.0)
    parser.add_argument("--severity_sigma", type=float, default=0.9)
    parser.add_argument("--premium", type=float, default=7000.0)
    parser.add_argument("--expenses", type=float, default=1200000.0)
    parser.add_argument("--policy_limit", type=float, default=200000.0)
    parser.add_argument("--deductible", type=float, default=0.0)
    parser.add_argument("--capital_buffer", type=float, default=2500000.0)
    parser.add_argument("--n_simulations", type=int, default=10000)
    parser.add_argument("--freq_stress", type=float, default=1.0)
    parser.add_argument("--sev_stress", type=float, default=1.0)
    parser.add_argument("--scenario_name", type=str, default="Custom Scenario")
    args = parser.parse_args()

    config = PortfolioConfig(
        n_policies=args.n_policies,
        expected_claim_frequency_per_policy=args.claim_freq,
        avg_claim_severity=args.avg_severity,
        severity_sigma=args.severity_sigma,
        premium_per_policy=args.premium,
        fixed_expenses=args.expenses,
        policy_limit=args.policy_limit,
        deductible=args.deductible,
    )

    scenario = StressScenario(
        frequency_multiplier=args.freq_stress,
        severity_multiplier=args.sev_stress,
        name=args.scenario_name,
    )

    simulator = InsuranceRiskSimulator(config)
    results = simulator.run_simulation(args.n_simulations, scenario)
    summary = summarize_results(results, capital_buffer=args.capital_buffer)
    print(json.dumps({"scenario": scenario.name, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
