#!/usr/bin/env python3
"""
Algebra EBM Experiment Runner

Orchestrates and executes the comprehensive evaluation suite for Algebra EBM models.
Reads pipeline.json to determine experiments to run and manages execution state.

Usage:
    python run_experiments.py                    # Run all READY experiments
    python run_experiments.py --experiment exp_001_single_rule_baseline
    python run_experiments.py --sequential       # Run one at a time
    python run_experiments.py --quick-test       # Small dataset for quick validation
"""

import argparse
import json
import logging
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment_runner.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages execution of Algebra EBM experiments."""

    def __init__(self, project_dir: Optional[Path] = None):
        """Initialize experiment runner.

        Args:
            project_dir: Path to algebra-ebm project directory
        """
        self.project_dir = project_dir if project_dir is not None else Path(__file__).parent
        self.pipeline_path = self.project_dir / '.state' / 'pipeline.json'
        self.runs_dir = self.project_dir / 'runs'
        self.runs_dir.mkdir(exist_ok=True)

        self.pipeline = self._load_pipeline()
        self.current_run_id = None

    def _load_pipeline(self) -> Dict[str, Any]:
        """Load pipeline configuration from JSON."""
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {self.pipeline_path}")

        with open(self.pipeline_path, 'r') as f:
            return json.load(f)

    def _save_pipeline(self):
        """Save updated pipeline configuration."""
        with open(self.pipeline_path, 'w') as f:
            json.dump(self.pipeline, f, indent=2)

    def get_ready_experiments(self) -> List[Dict[str, Any]]:
        """Get list of experiments ready to run."""
        return [exp for exp in self.pipeline.get('pending_experiments', [])
                if exp['status'] == 'READY']

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment configuration by ID."""
        for exp in self.pipeline.get('pending_experiments', []):
            if exp['experiment_id'] == experiment_id:
                return exp
        return None

    def create_run_directory(self, experiment_id: str) -> Path:
        """Create directory for experiment run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.runs_dir / f"{experiment_id}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / 'results').mkdir(exist_ok=True)
        (run_dir / 'logs').mkdir(exist_ok=True)

        return run_dir

    def run_experiment(self, experiment: Dict[str, Any], run_dir: Path) -> bool:
        """Execute a single experiment.

        Args:
            experiment: Experiment configuration
            run_dir: Directory to save results

        Returns:
            True if successful, False otherwise
        """
        experiment_id = experiment['experiment_id']
        config = experiment['config']

        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {experiment_id}")
        logger.info(f"Description: {experiment['description']}")
        logger.info(f"Results: {run_dir}/results/")
        logger.info(f"{'='*70}\n")

        # Build command
        cmd = ['python', 'eval_algebra.py']

        # Add configuration arguments
        if 'eval_type' in config:
            cmd.extend(['--eval_type', config['eval_type']])

        if 'rules' in config:
            for rule in config['rules']:
                cmd.extend(['--rule', rule])

        if 'num_rules' in config:
            cmd.extend(['--num_rules', str(config['num_rules'])])

        if 'num_problems' in config:
            cmd.extend(['--single_rule_problems', str(config['num_problems'])])
            cmd.extend(['--multi_rule_problems', str(config['num_problems'])])
            cmd.extend(['--constrained_problems', str(config['num_problems'])])

        if 'max_samples' in config:
            cmd.extend(['--max_samples', str(config['max_samples'])])

        if 'seed' in config:
            cmd.extend(['--seed', str(config['seed'])])

        # Output directory
        cmd.extend(['--output_dir', str(run_dir / 'results')])

        # Save detailed results
        cmd.append('--save_detailed')

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.project_dir}")

        # Run experiment
        try:
            start_time = time.time()

            log_file = run_dir / 'logs' / f'{experiment_id}.log'
            with open(log_file, 'w') as log_f:  # noqa: F841
                result = subprocess.run(
                    cmd,
                    cwd=self.project_dir,
                    capture_output=False,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✓ {experiment_id} completed successfully in {elapsed_time:.1f}s")
                return True
            else:
                logger.error(f"✗ {experiment_id} failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {experiment_id} timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"✗ {experiment_id} failed with exception: {str(e)}")
            return False

    def run_all(self, experiments: List[Dict[str, Any]], sequential: bool = False) -> Dict[str, bool]:
        """Run multiple experiments.

        Args:
            experiments: List of experiment configurations
            sequential: If True, run one at a time; if False, run in parallel

        Returns:
            Dictionary mapping experiment_id to success status
        """
        results = {}

        for exp in experiments:
            exp_id = exp['experiment_id']
            run_dir = self.create_run_directory(exp_id)

            success = self.run_experiment(exp, run_dir)
            results[exp_id] = success

            # Update pipeline
            exp['status'] = 'COMPLETED' if success else 'FAILED'
            self._save_pipeline()

            if not sequential:
                logger.info(f"Continuing to next experiment...")

            time.sleep(1)  # Brief pause between experiments

        return results

    def print_summary(self, results: Dict[str, bool]):
        """Print execution summary."""
        logger.info(f"\n{'='*70}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}\n")

        completed = sum(1 for v in results.values() if v)
        total = len(results)

        logger.info(f"Completed: {completed}/{total}")

        for exp_id, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"  {status}: {exp_id}")

        logger.info(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Algebra EBM Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                      # Run all READY experiments
  python run_experiments.py --experiment exp_001 # Run specific experiment
  python run_experiments.py --sequential         # Run one at a time
  python run_experiments.py --quick-test         # Small validation test
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Specific experiment ID to run (default: run all READY experiments)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run experiments sequentially instead of parallel'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick validation test with small dataset'
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        help='Path to algebra-ebm project directory'
    )

    args = parser.parse_args()

    try:
        # Initialize runner
        runner = ExperimentRunner(args.project_dir)
        logger.info(f"Loaded pipeline from {runner.pipeline_path}")

        # Determine which experiments to run
        if args.quick_test:
            exp = runner.get_experiment('exp_006_quick_validation')
            experiments = [exp] if exp else []
        elif args.experiment:
            exp = runner.get_experiment(args.experiment)
            if not exp:
                logger.error(f"Experiment not found: {args.experiment}")
                sys.exit(1)
            experiments = [exp]
        else:
            experiments = runner.get_ready_experiments()
            if not experiments:
                logger.warning("No READY experiments found in pipeline")
                sys.exit(0)

        logger.info(f"Found {len(experiments)} experiment(s) to run")

        # Run experiments
        results = runner.run_all(experiments, sequential=args.sequential)

        # Print summary
        runner.print_summary(results)

        # Exit with appropriate code
        if all(results.values()):
            logger.info("All experiments completed successfully!")
            sys.exit(0)
        else:
            logger.error("Some experiments failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
