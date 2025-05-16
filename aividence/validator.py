"""
ClaimValidator: Core functionality to validate scientific claims against literature
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .utils.pubmed import generate_search_terms, fetch_literature_parallel
from .utils.embedding import find_relevant_papers
from .utils.visualization import visualize_results, generate_report

class ClaimValidator:
    """
    Framework for validating scientific claims against real scientific literature.
    """
    
    def __init__(
        self, 
        email: str,
        llm_api_key: str,
        model_path: str = None,
        llm_base_url: str = "https://api.anthropic.com/v1",
        llm_model: str = "claude-3-opus-20240229",
        cache_dir: str = "validation_cache",
        max_workers: int = 5
    ):
        """
        Initialize the claim validator.
        
        Args:
            email: Email for PubMed API
            llm_api_key: API key for LLM service
            model_path: Path to local embedding model (default: sentence-transformers/all-mpnet-base-v2)
            llm_base_url: Base URL for LLM service
            llm_model: Model to use for evaluation
            cache_dir: Directory to cache results
            max_workers: Maximum number of parallel workers for PubMed searches
        """
        self.email = email
        
        # Setup embedding model
        self.model_path = model_path or "sentence-transformers/all-mpnet-base-v2"
        print(f"Loading embedding model from {self.model_path}...")
        self.embedding_model = SentenceTransformer(self.model_path)
        
        # Setup LLM client
        self.llm_client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )
        self.llm_model = llm_model
        
        # Create cache directory if it doesn't exist
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Set parallel workers
        self.max_workers = max_workers
            
        # Initialize tracking variables
        self.claim = None
        self.abstracts = []
        self.metadata = []
        self.evaluations = []
        self.distances = []
        self.indices = []
        self.summary = None
            
    def validate_claim(
        self, 
        claim: str,
        search_terms: Optional[List[str]] = None,
        max_papers_to_analyze: int = 30,
        show_plot: bool = True
    ) -> Dict:
        """
        Validate a scientific claim against the literature.
        
        Args:
            claim: Scientific claim to validate
            search_terms: Optional list of search terms. If None, will generate from claim
            max_papers_to_analyze: Maximum number of papers to analyze in detail
            show_plot: Whether to display plots directly
            
        Returns:
            Dictionary with validation results
        """
        self.claim = claim
        print(f"Validating claim: {claim}")
        
        # Generate search terms if not provided
        if search_terms is None:
            print("Generating search terms...")
            search_terms = generate_search_terms(self.llm_client, self.llm_model, claim)
            print(f"Generated {len(search_terms)} search terms: {', '.join(search_terms)}")
            
        # Fetch literature in parallel
        self.abstracts, self.metadata = fetch_literature_parallel(
            search_terms, 
            self.email, 
            self.cache_dir, 
            max_workers=self.max_workers
        )
        
        # If no abstracts found, return early
        if not self.abstracts:
            return {
                "claim": claim,
                "validity": "unknown",
                "confidence": 0.0,
                "message": "No relevant literature found to validate claim",
                "timestamp": datetime.now().isoformat()
            }
        
        # Find most relevant papers
        self.indices, self.distances = find_relevant_papers(
            self.abstracts, 
            claim, 
            self.embedding_model, 
            max_papers=max_papers_to_analyze
        )
        
        # Evaluate relevant papers
        self._evaluate_papers()
        
        # Synthesize results
        self._synthesize_findings()
        
        # Display results if requested
        if show_plot:
            fig = self.visualize_results()
            plt.show()
            
        return self.summary
    
    def _evaluate_papers(self) -> None:
        """
        Evaluate the stance of each relevant paper towards the claim.
        """
        if not hasattr(self, 'indices') or len(self.indices) == 0:
            print("No relevant papers found for evaluation")
            return
            
        print("\nEvaluating papers against claim...")
        evaluations = []
        
        # 使用tqdm为评估过程添加进度条
        for i, idx in tqdm(enumerate(self.indices), total=len(self.indices),
                          desc="Evaluating papers", unit="paper"):
            abstract = self.abstracts[idx]
            metadata = self.metadata[idx]
            
            prompt = f"""You are evaluating whether a scientific paper supports or refutes a claim.

CLAIM: {self.claim}

PAPER ABSTRACT: {abstract}

Assess whether this abstract supports, refutes, or is neutral toward the claim. Consider:
1. Directness of evidence (does it directly address the claim?)
2. Research methodology (is the evidence strong?)
3. Consistency with the claim's mechanism
4. Conflicts of interest or limitations

First, provide your stance as exactly one of these words: SUPPORT, REFUTE, or NEUTRAL.
Then on a new line, provide your confidence level as a number between 0 and 1.
Then on a new line, provide your reasoning.
Then on a new line, provide the evidence quality as: STRONG, MODERATE, or WEAK.
Then on a new line, provide the relevance as: HIGH, MEDIUM, or LOW.
"""

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                
                # Parse text response
                response_text = response.choices[0].message.content.strip().split('\n')
                
                # Set defaults
                stance = "neutral"
                confidence = 0.5
                reasoning = ""
                evidence_quality = "moderate"
                relevance = "medium"
                
                # Parse stance from first line
                if len(response_text) > 0:
                    first_line = response_text[0].upper()
                    if "SUPPORT" in first_line:
                        stance = "support"
                    elif "REFUTE" in first_line:
                        stance = "refute"
                    else:
                        stance = "neutral"
                
                # Try to extract confidence from second line
                if len(response_text) > 1:
                    try:
                        import re
                        confidence_matches = re.findall(r"0\.\d+|\d+\.\d+|\d+", response_text[1])
                        if confidence_matches:
                            confidence = float(confidence_matches[0])
                            # Ensure within 0-1 range
                            confidence = max(0, min(confidence, 1))
                    except:
                        pass
                
                # Extract reasoning and other metrics
                if len(response_text) > 2:
                    reasoning = " ".join(response_text[2:])
                    
                    # Try to extract evidence quality
                    if "STRONG" in reasoning.upper():
                        evidence_quality = "strong"
                    elif "WEAK" in reasoning.upper():
                        evidence_quality = "weak"
                    else:
                        evidence_quality = "moderate"
                        
                    # Try to extract relevance
                    if "HIGH" in reasoning.upper():
                        relevance = "high"
                    elif "LOW" in reasoning.upper():
                        relevance = "low"
                    else:
                        relevance = "medium"
                
                result = {
                    "stance": stance,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "evidence_quality": evidence_quality,
                    "relevance": relevance,
                    "consistency": 5  # Default medium consistency
                }
                
                evaluations.append(result)
                
            except Exception as e:
                print(f"  Error in evaluation: {str(e)[:100]}...")
                evaluations.append({
                    "stance": "neutral",
                    "confidence": 0.5,
                    "reasoning": f"Evaluation error: {str(e)[:100]}",
                    "evidence_quality": "weak",
                    "relevance": "low",
                    "consistency": 3
                })
                    
            # Small delay to avoid rate limits
            import time
            time.sleep(0.5)
            
        self.evaluations = evaluations
        
    def _synthesize_findings(self) -> None:
        """
        Synthesize the evaluation results into an overall assessment.
        """
        if not self.evaluations:
            print("No evaluations available for synthesis")
            self.summary = {
                "claim": self.claim,
                "validity": "unknown",
                "confidence": 0.0,
                "message": "No evaluations available",
                "timestamp": datetime.now().isoformat()
            }
            return
            
        print("\nSynthesizing findings...")
            
        # Count stances with progress bar
        stances = [e.get("stance", "neutral") for e in self.evaluations]
        stance_counts = {
            "support": stances.count("support"),
            "refute": stances.count("refute"),
            "neutral": stances.count("neutral")
        }
        
        print(f"Evidence distribution: Support: {stance_counts['support']}, " + 
              f"Refute: {stance_counts['refute']}, Neutral: {stance_counts['neutral']}")
        
        # Calculate weighted scores based on confidence and evidence quality
        support_score = 0
        refute_score = 0
        total_weight = 0
        
        # 添加进度条用于权重计算过程
        print("Calculating weighted scores...")
        for eval_result in tqdm(self.evaluations, desc="Processing evaluations"):
            if "stance" not in eval_result:
                continue
                
            # Calculate weight based on evidence quality and relevance
            quality_weight = {
                "strong": 1.0,
                "moderate": 0.7,
                "weak": 0.3
            }.get(eval_result.get("evidence_quality", "moderate"), 0.5)
            
            relevance_weight = {
                "high": 1.0,
                "medium": 0.7,
                "low": 0.3
            }.get(eval_result.get("relevance", "medium"), 0.5)
            
            confidence = eval_result.get("confidence", 0.5)
            
            # Combined weight
            weight = quality_weight * relevance_weight * confidence
            total_weight += weight
            
            # Add to appropriate score
            if eval_result["stance"] == "support":
                support_score += weight
            elif eval_result["stance"] == "refute":
                refute_score += weight
        
        # Normalize scores
        if total_weight > 0:
            support_score /= total_weight
            refute_score /= total_weight
            
        # Calculate overall validity score (-1 to 1 scale)
        validity_score = support_score - refute_score
        
        # Convert to validity assessment
        if validity_score > 0.3:
            validity = "supported"
            confidence = min(validity_score + 0.5, 1.0)  # Scale to 0.5-1.0 range
        elif validity_score < -0.3:
            validity = "refuted"
            confidence = min(abs(validity_score) + 0.5, 1.0)  # Scale to 0.5-1.0 range
        else:
            validity = "inconclusive"
            confidence = 0.5 - abs(validity_score * 0.5)  # Higher for more neutral
            
        print(f"Validity assessment: {validity.capitalize()} (confidence: {confidence:.2f})")
        
        # Get detailed synthesis from LLM
        print("Generating detailed synthesis...")
        synthesis_prompt = f"""As a scientific evaluator, assess the validity of this claim based on literature evidence:

CLAIM: {self.claim}

EVIDENCE SUMMARY:
- Supporting studies: {stance_counts['support']}
- Refuting studies: {stance_counts['refute']}
- Neutral studies: {stance_counts['neutral']}
- Weighted support score: {support_score:.2f}
- Weighted refute score: {refute_score:.2f}

Based on this evidence, provide:
1. A verdict on whether the claim is supported, refuted, or inconclusive
2. An explanation of your reasoning
3. Discussion of the strength and limitations of evidence
4. Assessment of whether this claim appears scientifically valid

Keep your response concise but comprehensive.
"""

        try:
            with tqdm(total=1, desc="Generating synthesis") as pbar:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.3
                )
                pbar.update(1)
            
            llm_synthesis = response.choices[0].message.content
        except Exception as e:
            llm_synthesis = f"Unable to generate synthesis: {str(e)[:100]}"
            
        # Create overall summary
        self.summary = {
            "claim": self.claim,
            "validity": validity,
            "confidence": confidence,
            "support_score": support_score,
            "refute_score": refute_score,
            "stance_counts": stance_counts,
            "total_papers_analyzed": len(self.evaluations),
            "total_papers_retrieved": len(self.abstracts),
            "llm_synthesis": llm_synthesis,
            "timestamp": datetime.now().isoformat()
        }
        
        print("Validation complete!")
    
    def visualize_results(self) -> plt.Figure:
        """
        Create visualization of validation results.
        
        Returns:
            Matplotlib figure
        """
        print("Generating visualizations...")
        with tqdm(total=1, desc="Creating plots") as pbar:
            fig = visualize_results(self.summary, self.evaluations, self.metadata, self.indices)
            pbar.update(1)
        return fig
        
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive report of the validation results.
        
        Args:
            output_file: Optional file to save the report to
            
        Returns:
            Report text
        """
        print("Generating report...")
        with tqdm(total=1, desc="Creating report") as pbar:
            report = generate_report(
                self.claim,
                self.summary,
                self.evaluations,
                self.metadata,
                self.indices,
                output_file
            )
            pbar.update(1)
            
        if output_file:
            print(f"Report saved to: {output_file}")
            
        return report