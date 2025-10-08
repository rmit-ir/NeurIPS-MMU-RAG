#!/usr/bin/env python3
"""
Evaluation script for RAG systems following G-RAG-LiveRAG pattern.

This script:
1. Takes a RAG system result file (JSONL)
2. Takes a reference dataset with QA pairs  
3. Runs the specified evaluator on the results
4. Saves evaluation results to files

Usage examples:
    # RAGAS evaluation
    python scripts/evaluate.py \\
        --evaluator RAGASEvaluator \\
        --results data/system_outputs.jsonl \\
        --reference data/references.jsonl \\
        --model-name openai/gpt-4o-mini \\
        --api-key sk-or-v1-your-key
    
    # NLP metrics evaluation  
    python scripts/evaluate.py \\
        --evaluator NLPMetricsEvaluator \\
        --results data/system_outputs.jsonl \\
        --reference data/references.jsonl
"""

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union
from typing import Dict, List, Any, Type

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluators.evaluator_interface import EvaluatorInterface


def find_evaluator_classes() -> Dict[str, str]:
    """
    Find all available evaluator classes in the evaluators directory.
    
    Returns:
        Dictionary mapping evaluator class names to their module paths
    """
    evaluators = {}
    evaluators_dir = project_root / "src" / "evaluators"
    
    for item in evaluators_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__"):
            # Look for evaluator.py in subdirectories
            evaluator_file = item / "evaluator.py"
            if evaluator_file.exists():
                # Convert path to module format
                module_path = f"src.evaluators.{item.name}.evaluator"
                
                # Try to import and find evaluator classes
                try:
                    module = importlib.import_module(module_path)
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (inspect.isclass(obj) and 
                            issubclass(obj, EvaluatorInterface) and 
                            obj != EvaluatorInterface):
                            evaluators[name] = f"{module_path}.{name}"
                except ImportError:
                    continue
    
    return evaluators


def load_evaluator_class(evaluator_path: str) -> Type[EvaluatorInterface]:
    """
    Load an evaluator class from a module path.
    
    Args:
        evaluator_path: Either full module path or class name
        
    Returns:
        The evaluator class
    """
    if '.' in evaluator_path:
        # Full module path provided
        module_path, class_name = evaluator_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # Class name only - find it
        available_evaluators = find_evaluator_classes()
        if evaluator_path not in available_evaluators:
            available_names = list(available_evaluators.keys())
            raise ValueError(f"Evaluator '{evaluator_path}' not found. Available: {available_names}")
        
        return load_evaluator_class(available_evaluators[evaluator_path])


def extract_evaluator_parameters(evaluator_class: Type[EvaluatorInterface]) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameter information from evaluator's __init__ method.
    
    Args:
        evaluator_class: The evaluator class to inspect
        
    Returns:
        Dictionary with parameter info (type, default, description)
    """
    signature = inspect.signature(evaluator_class.__init__)
    docstring = evaluator_class.__init__.__doc__ or ""
    
    # Parse parameter descriptions from docstring
    param_descriptions = {}
    current_param = None
    
    for line in docstring.split('\n'):
        line = line.strip()
        if line.startswith('Args:'):
            continue
        
        # Look for parameter documentation
        if ': ' in line and not line.startswith(' '):
            parts = line.split(': ', 1)
            if len(parts) == 2:
                current_param = parts[0].strip()
                param_descriptions[current_param] = parts[1].strip()
        elif current_param and line:
            param_descriptions[current_param] += ' ' + line
    
    # Extract parameter info
    params = {}
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
            
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        default = param.default if param.default != inspect.Parameter.empty else None
        description = param_descriptions.get(name, f"Parameter '{name}'")
        
        params[name] = {
            'type': param_type,
            'default': default,
            'description': description
        }
    
    return params


def load_system_outputs(file_path: str) -> List[Dict[str, Any]]:
    """Load system outputs from JSONL file."""
    outputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))
    return outputs


def load_references(file_path: str) -> List[Dict[str, Any]]:
    """Load reference data from JSON or JSONL file."""
    references = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # Try to parse as JSON array first
    if content.startswith('[') and content.endswith(']'):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                references = data
            else:
                references = [data]
        except json.JSONDecodeError:
            pass
    
    # If not a JSON array, try parsing as JSONL
    if not references:
        references = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    references.append(json.loads(line))
    
    return references


def save_evaluation_results(
    result,
    base_name: str,
    output_format: str = 'jsonl'
) -> None:
    """
    Save evaluation results to files.
    
    Args:
        result: EvaluationResult object
        base_name: Base filename without extension
        output_format: 'jsonl' or 'tsv'
    """
    ext = '.jsonl' if output_format == 'jsonl' else '.tsv'
    
    # Save aggregated results
    agg_file = f"{base_name}.aggregated{ext}"
    
    if output_format == 'jsonl':
        agg_data = result.to_dict()
        agg_data.pop('rows', None)  # Remove rows from aggregated file
        
        with open(agg_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(agg_data) + '\n')
    else:
        # TSV format for aggregated results
        import pandas as pd
        
        agg_dict = {
            'evaluator_name': result.evaluator_name,
            'sample_count': result.sample_count,
            'timestamp': result.timestamp.isoformat(),
            **result.metrics
        }
        
        if result.total_time_ms:
            agg_dict['total_time_ms'] = result.total_time_ms
        
        df = pd.DataFrame([agg_dict])
        df.to_csv(agg_file, sep='\\t', index=False)
    
    # Save row-level results if available
    if result.rows:
        rows_file = f"{base_name}.rows{ext}"
        
        if output_format == 'jsonl':
            with open(rows_file, 'w', encoding='utf-8') as f:
                for row in result.rows:
                    f.write(json.dumps(row) + '\n')
        else:
            import pandas as pd
            df = pd.DataFrame(result.rows)
            df.to_csv(rows_file, sep='\\t', index=False)
        
        print(f"Results saved:")
        print(f"  - Aggregated: {agg_file}")
        print(f"  - Row-level: {rows_file}")
    else:
        print(f"Results saved: {agg_file}")


def create_parser(evaluator_class=None):
    """Create argument parser with optional evaluator-specific parameters."""
    if evaluator_class:
        description = f"Evaluate RAG results using {evaluator_class.__name__}"
        if evaluator_class.__doc__:
            description += f"\\n\\n{evaluator_class.__doc__.strip()}"
    else:
        description = "Evaluate RAG system results against reference answers"
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core arguments
    parser.add_argument(
        '--evaluator', 
        type=str,
        default='RAGASEvaluator',
        help='Evaluator to use (class name or full path)'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to system outputs file (JSONL format)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help='Path to reference data file (JSONL format)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/evaluation_results)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Output filename prefix (default: based on evaluator name)'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['jsonl', 'tsv'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    
    # Add evaluator-specific parameters
    if evaluator_class:
        eval_params = extract_evaluator_parameters(evaluator_class)
        
        for name, param_info in eval_params.items():
            arg_name = f'--{name.replace("_", "-")}'
            param_type = param_info['type']
            
            if param_type == bool:
                # Boolean parameters get --param and --no-param flags
                parser.add_argument(
                    arg_name,
                    action='store_true',
                    dest=name,
                    help=f"{param_info['description']} (default: {param_info['default']})"
                )
                parser.add_argument(
                    f'--no-{name.replace("_", "-")}',
                    action='store_false',
                    dest=name,
                    help=f"Disable {name}"
                )
                parser.set_defaults(**{name: param_info['default']})
            else:
                # Regular parameters - handle Optional types specially
                if (hasattr(param_type, '__origin__') and 
                    param_type.__origin__ is Union and 
                    len(param_type.__args__) == 2 and 
                    type(None) in param_type.__args__):
                    # This is Optional[SomeType] - use the non-None type
                    actual_type = next((t for t in param_type.__args__ if t is not type(None)), str)
                    parser.add_argument(
                        arg_name,
                        type=actual_type,
                        default=param_info['default'],
                        help=f"{param_info['description']} (default: {param_info['default']})"
                    )
                else:
                    # Regular type
                    parser.add_argument(
                        arg_name,
                        type=param_type,
                        default=param_info['default'],
                        help=f"{param_info['description']} (default: {param_info['default']})"
                    )
    
    return parser


def main():
    """Main entry point."""
    start_time = time.time()
    
    # First pass: get evaluator name to load specific parameters
    basic_parser = argparse.ArgumentParser(add_help=False)
    basic_parser.add_argument('--evaluator', type=str, default='RAGASEvaluator')
    basic_args, _ = basic_parser.parse_known_args()
    
    # Load evaluator class and create full parser
    try:
        evaluator_class = load_evaluator_class(basic_args.evaluator)
        parser = create_parser(evaluator_class)
    except Exception as e:
        print(f"Warning: Could not load evaluator '{basic_args.evaluator}': {e}")
        print("Available evaluators:")
        for name in find_evaluator_classes():
            print(f"  - {name}")
        sys.exit(1)
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        output_dir = project_root / "data" / "evaluation_results"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    results_name = Path(args.results).stem
    timestamp = datetime.now().strftime("%m%d%H%M")
    evaluator_name = args.evaluator.split('.')[-1] if '.' in args.evaluator else args.evaluator
    
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        prefix = f"{results_name}.eval{timestamp}.{evaluator_name}"
    
    base_name = output_dir / prefix
    
    try:
        # Load evaluator class and extract parameters
        evaluator_class = load_evaluator_class(args.evaluator)
        
        # Get evaluator parameters from args
        eval_params = extract_evaluator_parameters(evaluator_class)
        evaluator_kwargs = {}
        
        for param_name in eval_params:
            if hasattr(args, param_name):
                value = getattr(args, param_name)
                if value is not None:
                    evaluator_kwargs[param_name] = value
        
        # Initialize evaluator
        evaluator = evaluator_class(**evaluator_kwargs)
        
        print(f"🔄 Loading data...")
        # Load data
        system_outputs = load_system_outputs(args.results)
        references = load_references(args.reference)
        
        print(f"📊 Running {evaluator.name}...")
        print(f"   {evaluator.description}")
        
        # Run evaluation
        result = evaluator.evaluate(system_outputs, references)
        
        # Save results
        print(f"💾 Saving results...")
        save_evaluation_results(result, str(base_name), args.output_format)
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\\n✅ Evaluation completed in {total_time:.1f}s")
        print(f"📈 Results for {result.sample_count} samples:")
        
        for metric, value in result.metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        if result.total_time_ms:
            print(f"⏱️  Evaluation time: {result.total_time_ms/1000:.1f}s")
            print(f"   Average per sample: {result.total_time_ms/result.sample_count:.0f}ms")
    
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
