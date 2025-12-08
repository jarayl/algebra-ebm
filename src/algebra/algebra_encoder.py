"""
Algebraic Equation Encoder

Implements encoding of symbolic algebra strings into continuous embeddings.
Provides both character-level baseline and AST-based encoders using SymPy.

Example Usage:
    # Character-level encoding
    char_encoder = create_character_encoder(d_model=128)
    embedding = char_encoder("2*x+3=7")
    
    # AST-based encoding  
    ast_encoder = create_ast_encoder(d_model=128)
    embedding = ast_encoder("2*x+3=7")
    
    # Decoding (requires sklearn: pip install scikit-learn)
    decoder = create_decoder_with_default_candidates(char_encoder)
    decoded_eq, distance = decoder.decode_embedding(embedding)
    
    # Validation and testing
    is_valid, error, expr = validate_equation_syntax("2*x+3=7") 
    solutions, error = solve_equation("2*x+3=7")
    equiv, error = check_equation_equivalence("2*x+3=7", "2*x=4")
"""

import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from typing import Union, List, Optional, Dict, Any


def _validate_simple_expression(expr_str: str) -> None:
    """
    Simple expression safety validation for global functions.
    
    Blocks dangerous patterns that could execute code while allowing mathematical expressions.
    """
    dangerous_patterns = [
        '__import__', '__builtins__', '__globals__', '__locals__',
        'eval', 'exec', 'compile', 'open', 'input', 'raw_input',
        'file', 'execfile', 'reload', 'import', 'from ',
        'os.', 'sys.', 'subprocess', 'shutil', 'tempfile'
    ]
    
    expr_lower = expr_str.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            raise ValueError(f"Expression contains potentially unsafe pattern: {pattern}")


class CharacterLevelEncoder(nn.Module):
    """
    Character-level encoder for algebraic equations.
    
    Converts equation strings into continuous embeddings using one-hot encoding
    per character, flattened and projected to d_model dimensions.
    
    Args:
        d_model: Output embedding dimension (default: 128)
        max_len: Maximum sequence length with padding (default: 64)
        normalize_embeddings: Whether to normalize embeddings to unit norm (default: True)
    """
    
    def __init__(self, d_model: int = 128, max_len: int = 64, normalize_embeddings: bool = True):
        super().__init__()
        
        # Extended vocabulary to handle all equation generation formats
        # Includes decimal points, brackets, and extended characters that 
        # equation generation can produce (fixes BUG-5)
        self.vocab = '0123456789x.+-=*/()[]<> '
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = len(self.vocab)
        self.normalize_embeddings = normalize_embeddings
        
        # Codebase compatibility attributes
        self.inp_dim = d_model  # For interface compatibility
        self.out_dim = d_model  # For interface compatibility
        
        # Projection layer: (max_len * vocab_size) -> d_model
        input_dim = max_len * self.vocab_size
        self.projection = nn.Linear(input_dim, d_model)
        
        # Initialize projection weights with smaller scale for bounded outputs
        # Using orthogonal initialization helps maintain stable gradient flow
        nn.init.orthogonal_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
    
    def encode_equation_string(self, eq_str: str) -> torch.Tensor:
        """
        Encode a single equation string to embedding vector.
        
        Args:
            eq_str: Equation string (e.g., "2(x+3)+4=10")
            
        Returns:
            Tensor of shape (d_model,) containing the embedding
            
        Note:
            Embeddings are normalized to have unit L2 norm by default,
            which is important for the diffusion process that expects
            data in a bounded range.
        """
        # Get device from projection layer for consistency
        device = next(self.projection.parameters()).device
        
        # Create one-hot encoding tensor directly (more efficient)
        one_hot = torch.zeros((self.max_len, self.vocab_size), dtype=torch.float32, device=device)
        
        # Truncate or pad string to max_len
        eq_str = eq_str[:self.max_len]
        
        # One-hot encode each character
        for i, c in enumerate(eq_str):
            if c in self.char_to_idx:
                one_hot[i, self.char_to_idx[c]] = 1.0
            else:
                # Unknown character - could raise error or use space
                raise ValueError(f"Unknown character '{c}' not in vocabulary: {self.vocab}")
        
        # Flatten: (max_len, vocab_size) -> (max_len * vocab_size,)
        flat = one_hot.flatten()
        
        # Project to embedding space
        embedding = self.projection(flat)
        
        # CRITICAL FIX: Normalize embeddings for diffusion process
        # The diffusion model expects data roughly in [-1, 1] range
        # Normalizing to unit L2 norm ensures bounded, well-behaved embeddings
        if self.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def encode_batch(self, eq_strings: List[str]) -> torch.Tensor:
        """
        Encode a batch of equation strings.
        
        Args:
            eq_strings: List of equation strings
            
        Returns:
            Tensor of shape (batch_size, d_model)
        """
        batch_embeddings = []
        for eq_str in eq_strings:
            embedding = self.encode_equation_string(eq_str)
            batch_embeddings.append(embedding)
        
        return torch.stack(batch_embeddings, dim=0)
    
    def forward(self, eq_strings: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass - encode equation string(s).
        
        Args:
            eq_strings: Single string or list of strings
            
        Returns:
            Tensor of shape (d_model,) for single string or (batch_size, d_model) for batch
        """
        if isinstance(eq_strings, str):
            return self.encode_equation_string(eq_strings)
        else:
            return self.encode_batch(eq_strings)
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """Get vocabulary information."""
        return {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'd_model': self.d_model,
            'max_len': self.max_len
        }


class ASTEncoder(nn.Module):
    """
    AST-based encoder for algebraic equations using SymPy expression trees.
    
    Converts equation strings into continuous embeddings by parsing with SymPy,
    extracting features from the Abstract Syntax Tree structure, and projecting
    to d_model dimensions.
    
    Args:
        d_model: Output embedding dimension (default: 128)
        max_features: Maximum number of AST features to extract (default: 64)
    """
    
    def __init__(self, d_model: int = 128, max_features: int = 64):
        super().__init__()
        
        self.d_model = d_model
        self.max_features = max_features
        
        # Codebase compatibility attributes
        self.inp_dim = d_model
        self.out_dim = d_model
        
        # Define AST node type vocabulary for consistent encoding
        self.node_types = [
            'Symbol', 'Integer', 'Rational', 'Float', 'Add', 'Mul', 'Pow', 
            'Eq', 'Number', 'Atom', 'Function', 'Expr', 'Basic'
        ]
        self.node_type_to_idx = {nt: i for i, nt in enumerate(self.node_types)}
        
        # Projection layer: max_features -> d_model
        self.projection = nn.Linear(max_features, d_model)
        
        # Initialize projection weights
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def extract_ast_features(self, expr) -> List[float]:
        """
        Extract features from a SymPy expression's AST.
        
        Args:
            expr: SymPy expression
            
        Returns:
            List of float features representing the AST structure
        """
        features = []
        
        # Basic structure features
        features.append(float(len(expr.args)))  # Number of child nodes
        features.append(float(expr.func.__name__ in self.node_type_to_idx))  # Known node type
        
        # Node type encoding (one-hot style)
        node_type_name = expr.func.__name__
        for nt in self.node_types[:8]:  # Limit to avoid too many features
            features.append(float(node_type_name == nt))
        
        # Numerical features if it's a number
        if hasattr(expr, 'is_number') and expr.is_number:
            try:
                val = float(expr.evalf())
                features.extend([val, abs(val), val**2])
            except:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Symbolic features
        if hasattr(expr, 'free_symbols'):
            features.append(float(len(expr.free_symbols)))  # Number of variables
            features.append(float('x' in str(expr.free_symbols)))  # Contains x
        else:
            features.extend([0.0, 0.0])
        
        # Complexity features
        features.append(float(len(str(expr))))  # String length as complexity measure
        
        # Tree depth (iterative to avoid stack overflow)
        def get_depth(e):
            if not hasattr(e, 'args') or not e.args:
                return 1
            max_depth = 1
            stack = [(e, 1)]
            while stack:
                node, depth = stack.pop()
                max_depth = max(max_depth, depth)
                if hasattr(node, 'args') and node.args:
                    for arg in node.args:
                        stack.append((arg, depth + 1))
            return max_depth
        
        features.append(float(get_depth(expr)))
        
        # Pad or truncate to max_features
        while len(features) < self.max_features:
            features.append(0.0)
        features = features[:self.max_features]
        
        return features
    
    def _validate_expression_safety(self, expr_str: str) -> None:
        """
        Validate that expression string is safe for SymPy parsing.
        
        Blocks dangerous functions while allowing mathematical expressions.
        Throws ValueError if expression contains unsafe elements.
        """
        # List of dangerous patterns that could execute code
        dangerous_patterns = [
            '__import__', '__builtins__', '__globals__', '__locals__',
            'eval', 'exec', 'compile', 'open', 'input', 'raw_input',
            'file', 'execfile', 'reload', 'import', 'from ',
            'os.', 'sys.', 'subprocess', 'shutil', 'tempfile'
        ]
        
        expr_lower = expr_str.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                raise ValueError(f"Expression contains potentially unsafe pattern: {pattern}")
        
        # Additional check for parentheses with non-mathematical content
        if '(' in expr_str and ')' in expr_str:
            # Extract content within parentheses
            import re
            paren_contents = re.findall(r'\(([^)]+)\)', expr_str)
            for content in paren_contents:
                content_clean = content.strip().lower()
                # Allow mathematical function calls but block others
                if (content_clean and 
                    not re.match(r'^[a-z0-9\s+\-*/^.,]+$', content_clean) and
                    not any(func in content_clean for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt'])):
                    raise ValueError(f"Expression contains suspicious parenthetical content: {content}")
    
    def encode_equation_string(self, eq_str: str) -> torch.Tensor:
        """
        Encode a single equation string using AST features.
        
        Args:
            eq_str: Equation string (e.g., "2(x+3)+4=10")
            
        Returns:
            Tensor of shape (d_model,) containing the embedding
        """
        device = next(self.projection.parameters()).device
        
        try:
            # Parse equation with SymPy - handle multiple = signs
            if "=" in eq_str:
                eq_parts = eq_str.split("=")
                if len(eq_parts) != 2:
                    raise ValueError(f"Invalid equation format - expected exactly one '=' sign: {eq_str}")
                lhs_str, rhs_str = eq_parts[0].strip(), eq_parts[1].strip()
                # Safe sympify: evaluate=False prevents auto-simplification (correctness)
                # Input validation prevents code execution (security), rational=False for performance
                self._validate_expression_safety(lhs_str)
                self._validate_expression_safety(rhs_str)
                lhs = sp.sympify(lhs_str, evaluate=False, rational=False, convert_xor=False)
                rhs = sp.sympify(rhs_str, evaluate=False, rational=False, convert_xor=False)
                
                # For equations, extract features from both sides separately
                lhs_features = self.extract_ast_features(lhs)
                rhs_features = self.extract_ast_features(rhs)
                
                # Combine features by interleaving and averaging to maintain max_features size
                features = []
                half_features = self.max_features // 2
                
                # Take half features from each side
                for i in range(half_features):
                    if i < len(lhs_features):
                        features.append(lhs_features[i])
                    else:
                        features.append(0.0)
                        
                for i in range(half_features):
                    if i < len(rhs_features):
                        features.append(rhs_features[i])
                    else:
                        features.append(0.0)
                
                # Ensure exact max_features length
                features = features[:self.max_features]
                while len(features) < self.max_features:
                    features.append(0.0)
                    
            else:
                # No equals sign, treat as expression
                self._validate_expression_safety(eq_str.strip())
                expr = sp.sympify(eq_str.strip(), evaluate=False, rational=False, convert_xor=False)
                features = self.extract_ast_features(expr)
            
        except Exception as e:
            # Fallback for parsing errors - return zero features
            features = [0.0] * self.max_features
        
        # Convert to tensor and project
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        embedding = self.projection(features_tensor)
        
        return embedding
    
    def encode_batch(self, eq_strings: List[str]) -> torch.Tensor:
        """
        Encode a batch of equation strings.
        
        Args:
            eq_strings: List of equation strings
            
        Returns:
            Tensor of shape (batch_size, d_model)
        """
        batch_embeddings = []
        for eq_str in eq_strings:
            embedding = self.encode_equation_string(eq_str)
            batch_embeddings.append(embedding)
        
        return torch.stack(batch_embeddings, dim=0)
    
    def forward(self, eq_strings: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass - encode equation string(s) using AST features.
        
        Args:
            eq_strings: Single string or list of strings
            
        Returns:
            Tensor of shape (d_model,) for single string or (batch_size, d_model) for batch
        """
        if isinstance(eq_strings, str):
            return self.encode_equation_string(eq_strings)
        else:
            return self.encode_batch(eq_strings)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get AST feature extraction information."""
        return {
            'node_types': self.node_types,
            'max_features': self.max_features,
            'd_model': self.d_model,
            'feature_description': 'AST structure, node types, numerical values, complexity metrics'
        }


class EquationDecoder:
    """
    Reversible decoder for algebraic equations using nearest-neighbor search.
    
    Maintains a candidate set of valid equation strings with their embeddings,
    and decodes embeddings back to strings by finding the nearest neighbor.
    
    Args:
        encoder: Either CharacterLevelEncoder or ASTEncoder instance
        distance_threshold: Maximum distance for valid matches (default: 1.0)
    """
    
    def __init__(self, encoder: Union[CharacterLevelEncoder, ASTEncoder], distance_threshold: float = 1.0):
        self.encoder = encoder
        self.distance_threshold = distance_threshold
        
        # Will store candidate equations and their embeddings
        self.candidate_equations = []
        self.candidate_embeddings = None
        self.nn_search = None
        
        # Import sklearn here to avoid requiring it globally  
        try:
            from sklearn.neighbors import NearestNeighbors
            self.NearestNeighbors = NearestNeighbors
        except ImportError:
            raise ImportError(
                "sklearn is required for EquationDecoder. Install with: pip install scikit-learn\n"
                "Alternative: Use the encoders directly without decoding, or implement a custom decoder."
            )
    
    def build_candidate_set(self, equations: List[str], batch_size: int = 32):
        """
        Build the candidate set of equations and their embeddings.
        
        Args:
            equations: List of equation strings to use as candidates
            batch_size: Batch size for encoding (to manage memory)
        """
        if not equations:
            raise ValueError("Cannot build candidate set with empty equations list")
            
        self.candidate_equations = equations.copy()
        all_embeddings = []
        
        # Set encoder to eval mode to ensure consistent embeddings
        encoder_was_training = self.encoder.training
        self.encoder.eval()
        
        try:
            # Encode in batches to manage memory
            with torch.no_grad():  # Save memory and ensure deterministic results
                for i in range(0, len(equations), batch_size):
                    batch = equations[i:i+batch_size]
                    batch_embeddings = self.encoder.encode_batch(batch)
                    all_embeddings.append(batch_embeddings.detach().cpu())
            
            # Concatenate all embeddings - check for empty list
            if not all_embeddings:
                raise ValueError("No embeddings were generated from equations")
                
            self.candidate_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            
            # Build nearest neighbor search index
            self.nn_search = self.NearestNeighbors(
                n_neighbors=1, 
                metric='euclidean',
                algorithm='auto'
            )
            self.nn_search.fit(self.candidate_embeddings)
            
        finally:
            # Restore original training mode
            self.encoder.train(encoder_was_training)
    
    def build_default_candidate_set(self):
        """
        Build a default candidate set with common algebraic equations.
        Useful for testing and when no specific training set is available.
        """
        default_equations = [
            # Simple solved forms
            "x=0", "x=1", "x=2", "x=3", "x=4", "x=5", "x=-1", "x=-2",
            
            # Basic linear equations
            "x+1=2", "x+2=3", "x+3=4", "x-1=0", "x-2=1", "x-3=2",
            "2*x=4", "3*x=6", "4*x=8", "2*x=2", "3*x=3",
            
            # Distribution examples
            "2*(x+1)=4", "2*(x+2)=6", "3*(x+1)=6", "2*(x-1)=2",
            "2*x+2=4", "2*x+4=6", "3*x+3=6", "2*x-2=2",
            
            # Combination examples  
            "x+x=2", "2*x+x=6", "x+2*x=6", "3*x+2*x=10",
            "2*x+3*x=10", "x+x+x=6", "2*x+2*x=8",
            
            # More complex forms
            "2*(x+3)+4=10", "3*(x-1)+2=8", "4*(x+2)-3=13",
            "2*x+3*x+1=11", "x+2*(x+1)=7", "3*x-2*(x-1)=5",
            
            # Intermediate steps
            "2*x+6=10", "3*x-3=6", "4*x+8=16", "5*x-5=15",
            "6*x=12", "7*x=14", "8*x=16", "9*x=18", "10*x=20"
        ]
        
        self.build_candidate_set(default_equations)
    
    def decode_embedding(self, embedding: torch.Tensor) -> tuple:
        """
        Decode an embedding back to an equation string.
        
        Args:
            embedding: Tensor of shape (d_model,) containing the embedding
            
        Returns:
            Tuple of (equation_string, distance) where distance indicates match quality
        """
        if self.nn_search is None:
            raise ValueError("Candidate set not built. Call build_candidate_set() first.")
        
        # Convert to numpy if needed
        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.detach().cpu().numpy().reshape(1, -1)
        else:
            embedding_np = np.array(embedding).reshape(1, -1)
        
        # Find nearest neighbor
        distances, indices = self.nn_search.kneighbors(embedding_np)
        
        closest_idx = indices[0][0]
        distance = distances[0][0]
        
        # Check distance threshold
        if distance > self.distance_threshold:
            return None, distance  # No good match found
        
        return self.candidate_equations[closest_idx], distance
    
    def decode_batch(self, embeddings: torch.Tensor) -> List[tuple]:
        """
        Decode a batch of embeddings back to equation strings.
        
        Args:
            embeddings: Tensor of shape (batch_size, d_model)
            
        Returns:
            List of (equation_string, distance) tuples
        """
        results = []
        for embedding in embeddings:
            result = self.decode_embedding(embedding)
            results.append(result)
        return results
    
    def get_candidate_info(self) -> Dict[str, Any]:
        """Get information about the current candidate set."""
        return {
            'num_candidates': len(self.candidate_equations),
            'encoder_type': type(self.encoder).__name__,
            'distance_threshold': self.distance_threshold,
            'embedding_dim': self.candidate_embeddings.shape[1] if self.candidate_embeddings is not None else None
        }


# Helper Functions for Equation Validation and SymPy Integration

def validate_equation_syntax(eq_str: str) -> tuple:
    """
    Validate equation syntax and check if it can be parsed by SymPy.
    
    Args:
        eq_str: Equation string to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str or None, parsed_expr: sympy object or None)
    """
    try:
        eq_str = eq_str.strip()
        if not eq_str:
            return False, "Empty equation string", None
            
        # Check for basic syntax issues
        if eq_str.count('=') > 1:
            return False, "Multiple equals signs found", None
        
        # Try to parse with SymPy
        if "=" in eq_str:
            lhs_str, rhs_str = eq_str.split("=", 1)
            # Use a simple validation function for this global function
            _validate_simple_expression(lhs_str.strip())
            _validate_simple_expression(rhs_str.strip())
            lhs = sp.sympify(lhs_str.strip(), evaluate=False, rational=False, convert_xor=False)
            rhs = sp.sympify(rhs_str.strip(), evaluate=False, rational=False, convert_xor=False)
            expr = sp.Eq(lhs, rhs)
        else:
            _validate_simple_expression(eq_str)
            expr = sp.sympify(eq_str, evaluate=False, rational=False, convert_xor=False)
            
        return True, None, expr
        
    except (sp.SympifyError, ValueError, TypeError) as e:
        return False, f"SymPy parsing error: {str(e)}", None
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None


def solve_equation(eq_str: str, variable: str = 'x') -> tuple:
    """
    Solve an equation for a given variable using SymPy.
    
    Args:
        eq_str: Equation string (e.g., "2*x + 3 = 7")
        variable: Variable to solve for (default: 'x')
        
    Returns:
        Tuple of (solutions: list, error_message: str or None)
    """
    try:
        is_valid, error_msg, expr = validate_equation_syntax(eq_str)
        if not is_valid:
            return [], error_msg
            
        var_symbol = sp.Symbol(variable)
        
        if isinstance(expr, sp.Eq):
            # It's an equation
            solutions = sp.solve(expr, var_symbol)
        else:
            # It's an expression, assume equals zero
            solutions = sp.solve(expr, var_symbol)
            
        # Convert to float when possible for easier use
        numeric_solutions = []
        for sol in solutions:
            try:
                numeric_val = float(sol.evalf())
                numeric_solutions.append(numeric_val)
            except:
                numeric_solutions.append(sol)  # Keep symbolic if can't convert
                
        return numeric_solutions, None
        
    except Exception as e:
        return [], f"Error solving equation: {str(e)}"


def check_equation_equivalence(eq1_str: str, eq2_str: str, variable: str = 'x') -> tuple:
    """
    Check if two equations are mathematically equivalent.
    
    Args:
        eq1_str: First equation string
        eq2_str: Second equation string
        variable: Variable to check equivalence for (default: 'x')
        
    Returns:
        Tuple of (are_equivalent: bool, error_message: str or None)
    """
    try:
        # Solve both equations
        sol1, err1 = solve_equation(eq1_str, variable)
        sol2, err2 = solve_equation(eq2_str, variable)
        
        if err1:
            return False, f"Error in first equation: {err1}"
        if err2:
            return False, f"Error in second equation: {err2}"
            
        # Check if solution sets are the same
        if len(sol1) != len(sol2):
            return False, None
            
        # Compare solutions (handle both numeric and symbolic)
        try:
            # Sort solutions more robustly
            def robust_sort_key(x):
                if isinstance(x, (int, float)):
                    return (0, float(x))  # Numeric values first
                else:
                    return (1, str(x))  # Then symbolic values
            
            sol1_sorted = sorted(sol1, key=robust_sort_key)
            sol2_sorted = sorted(sol2, key=robust_sort_key)
            
            for s1, s2 in zip(sol1_sorted, sol2_sorted):
                if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                    if abs(s1 - s2) > 1e-10:  # Numerical tolerance
                        return False, None
                else:
                    # Symbolic comparison
                    if not sp.simplify(s1 - s2) == 0:
                        return False, None
            return True, None
            
        except Exception:
            # Fallback: try symbolic comparison (only if both have equals signs)
            if '=' in eq1_str and '=' in eq2_str:
                try:
                    # Safe sympify calls for equation comparison
                    lhs1, rhs1 = eq1_str.split('=')[0], eq1_str.split('=')[1]
                    lhs2, rhs2 = eq2_str.split('=')[0], eq2_str.split('=')[1]
                    _validate_simple_expression(lhs1)
                    _validate_simple_expression(rhs1)
                    _validate_simple_expression(lhs2)
                    _validate_simple_expression(rhs2)
                    eq1_lhs = sp.sympify(lhs1, evaluate=False, rational=False, convert_xor=False)
                    eq1_rhs = sp.sympify(rhs1, evaluate=False, rational=False, convert_xor=False)
                    eq2_lhs = sp.sympify(lhs2, evaluate=False, rational=False, convert_xor=False)
                    eq2_rhs = sp.sympify(rhs2, evaluate=False, rational=False, convert_xor=False)
                    diff1 = sp.simplify(eq1_lhs - eq1_rhs)
                    diff2 = sp.simplify(eq2_lhs - eq2_rhs)
                    return sp.simplify(diff1 - diff2) == 0, None
                except Exception:
                    pass
            return False, "Could not compare equations"
            
    except Exception as e:
        return False, f"Error checking equivalence: {str(e)}"


def create_character_encoder(d_model: int = 128, max_len: int = 64, normalize_embeddings: bool = True) -> CharacterLevelEncoder:
    """
    Factory function to create a CharacterLevelEncoder with standard settings.
    
    Args:
        d_model: Output embedding dimension
        max_len: Maximum sequence length
        normalize_embeddings: Whether to normalize embeddings to unit norm (default: True)
            This is important for diffusion training which expects bounded data.
        
    Returns:
        CharacterLevelEncoder instance
    """
    return CharacterLevelEncoder(d_model=d_model, max_len=max_len, normalize_embeddings=normalize_embeddings)


def create_ast_encoder(d_model: int = 128, max_features: int = 64) -> ASTEncoder:
    """
    Factory function to create an ASTEncoder with standard settings.
    
    Args:
        d_model: Output embedding dimension  
        max_features: Maximum number of AST features
        
    Returns:
        ASTEncoder instance
    """
    return ASTEncoder(d_model=d_model, max_features=max_features)


def create_decoder_with_default_candidates(encoder: Union[CharacterLevelEncoder, ASTEncoder], 
                                         distance_threshold: float = 1.0) -> EquationDecoder:
    """
    Factory function to create an EquationDecoder with default candidate set.
    
    Args:
        encoder: Encoder to use for decoding
        distance_threshold: Maximum distance for valid matches
        
    Returns:
        EquationDecoder instance with default candidates loaded
    """
    decoder = EquationDecoder(encoder, distance_threshold)
    decoder.build_default_candidate_set()
    return decoder


def create_decoder_from_dataset(encoder: Union[CharacterLevelEncoder, ASTEncoder],
                                dataset,
                                distance_threshold: float = 2.0,
                                include_inputs: bool = False) -> EquationDecoder:
    """
    Create an EquationDecoder with candidates from a dataset.
    
    This is CRITICAL for proper evaluation - the decoder needs to have
    the actual target equations from the dataset as candidates, not just
    a small fixed set of default equations.
    
    Args:
        encoder: Encoder to use for decoding
        dataset: AlgebraDataset or similar with get_equation_pair method
        distance_threshold: Maximum distance for valid matches (default 2.0 for normalized embeddings)
        include_inputs: If True, also include input equations as candidates
        
    Returns:
        EquationDecoder instance with dataset equations as candidates
    """
    decoder = EquationDecoder(encoder, distance_threshold)
    
    # Collect all unique target equations from dataset
    candidates = set()
    for i in range(len(dataset)):
        if hasattr(dataset, 'get_equation_pair'):
            input_eq, target_eq = dataset.get_equation_pair(i)
        elif hasattr(dataset, 'get_problem_info'):
            problem_info = dataset.get_problem_info(i)
            input_eq = problem_info['input_equation']
            target_eq = problem_info['target_equation']
        elif hasattr(dataset, 'equation_pairs'):
            input_eq, target_eq = dataset.equation_pairs[i]
        else:
            # Fallback: try to access as tuple
            input_eq, target_eq = dataset[i] if isinstance(dataset[i], tuple) else (None, None)
            if not isinstance(input_eq, str):
                raise ValueError("Dataset does not provide equation strings")
        
        candidates.add(target_eq)
        if include_inputs:
            candidates.add(input_eq)
    
    # Build candidate set
    candidates_list = sorted(list(candidates))  # Sort for reproducibility
    decoder.build_candidate_set(candidates_list)
    
    return decoder


def test_encoder_decoder_roundtrip(encoder, decoder, test_equations: List[str]) -> Dict[str, Any]:
    """
    Test encoder-decoder roundtrip accuracy on a set of equations.
    
    Args:
        encoder: Encoder instance
        decoder: Decoder instance  
        test_equations: List of equations to test
        
    Returns:
        Dictionary with test results and statistics
    """
    results = {
        'total_tests': len(test_equations),
        'successful_roundtrips': 0,
        'failed_roundtrips': 0,
        'average_distance': 0.0,
        'details': []
    }
    
    total_distance = 0.0
    
    for eq in test_equations:
        try:
            # Encode
            embedding = encoder.encode_equation_string(eq)
            
            # Decode  
            decoded_eq, distance = decoder.decode_embedding(embedding)
            
            # Check if successful
            if decoded_eq is not None:
                # Check equivalence
                are_equiv, equiv_error = check_equation_equivalence(eq, decoded_eq)
                if are_equiv and equiv_error is None:
                    results['successful_roundtrips'] += 1
                    success = True
                else:
                    results['failed_roundtrips'] += 1
                    success = False
            else:
                results['failed_roundtrips'] += 1
                success = False
                equiv_error = "No decoded equation found"
                
            total_distance += distance
            
            results['details'].append({
                'original': eq,
                'decoded': decoded_eq,
                'distance': distance,
                'success': success,
                'equiv_error': equiv_error if 'equiv_error' in locals() else None
            })
            
        except Exception as e:
            results['failed_roundtrips'] += 1
            results['details'].append({
                'original': eq,
                'decoded': None,
                'distance': float('inf'),
                'success': False,
                'error': str(e)
            })
    
    results['average_distance'] = total_distance / len(test_equations) if test_equations else 0.0
    results['success_rate'] = results['successful_roundtrips'] / len(test_equations) if test_equations else 0.0
    
    return results