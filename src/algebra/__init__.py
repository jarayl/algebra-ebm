# Algebra module - lazy imports to avoid circular dependencies
# Individual modules can be imported directly: from src.algebra.algebra_dataset import AlgebraDataset

__all__ = [
    # Dataset
    'AlgebraDataset',
    'MultiRuleDataset', 
    'ConstrainedDataset',
    # Encoder
    'CharacterLevelEncoder',
    'ASTEncoder',
    'EquationDecoder',
    'create_character_encoder',
    'create_ast_encoder',
    'create_decoder_with_default_candidates',
    'validate_equation_syntax',
    'check_equation_equivalence',
    'solve_equation',
    # Models
    'AlgebraEBM',
    'AlgebraDiffusionWrapper',
    'ContrastiveEnergyLoss',
    # Inference
    'AlgebraInference',
    'load_rule_models',
    'InferenceConfig',
    # Evaluation
    'evaluate_model_suite',
    'compute_embedding_distances',
]
