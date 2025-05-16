"""
Visualization utilities for claim validation results
"""

import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime  # 添加这一行导入 datetime 模块
from tqdm import tqdm

def visualize_results(
    summary: Dict, 
    evaluations: List[Dict], 
    metadata: List[Dict],
    indices: Optional[List[int]] = None
) -> plt.Figure:
    """
    Create visualization of validation results.
    
    Args:
        summary: Validation summary dict
        evaluations: List of paper evaluations
        metadata: List of paper metadata
        indices: Optional list of paper indices
        
    Returns:
        Matplotlib figure
    """
    if not evaluations:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No validation results available.", 
               ha='center', va='center')
        return fig
        
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    if 'claim' in summary:
        fig.suptitle(f"Validation Results: {summary['claim'][:70]}...", fontsize=16)
    else:
        fig.suptitle("Validation Results", fontsize=16)
    
    # 1. Stance distribution
    stances = [e.get("stance", "unknown") for e in evaluations]
    stance_counts = {
        "support": stances.count("support"),
        "refute": stances.count("refute"),
        "neutral": stances.count("neutral"),
        "unknown": stances.count("unknown")
    }
    
    # Remove zeros
    stance_counts = {k: v for k, v in stance_counts.items() if v > 0}
    
    colors = {
        "support": "green", 
        "refute": "red", 
        "neutral": "gray",
        "unknown": "lightgray"
    }
    
    axs[0, 0].bar(
        stance_counts.keys(), 
        stance_counts.values(),
        color=[colors[s] for s in stance_counts.keys()]
    )
    axs[0, 0].set_title('Evidence Stance Distribution')
    axs[0, 0].set_ylabel('Number of Papers')
    
    # 2. Evidence quality
    qualities = [e.get("evidence_quality", "unknown") for e in evaluations]
    quality_counts = {
        "strong": qualities.count("strong"),
        "moderate": qualities.count("moderate"),
        "weak": qualities.count("weak"),
        "unknown": qualities.count("unknown")
    }
    
    # Remove zeros
    quality_counts = {k: v for k, v in quality_counts.items() if v > 0}
    
    quality_colors = {
        "strong": "darkgreen", 
        "moderate": "orange", 
        "weak": "tomato", 
        "unknown": "lightgray"
    }
    
    axs[0, 1].bar(
        quality_counts.keys(),
        quality_counts.values(),
        color=[quality_colors[q] for q in quality_counts.keys()]
    )
    axs[0, 1].set_title('Evidence Quality Distribution')
    axs[0, 1].set_ylabel('Number of Papers')
    
    # 3. Publication years
    if indices is not None and len(indices) > 0:
        years = [metadata[idx].get("year") for idx in indices]
        years = [y for y in years if y is not None]
        
        if years:
            year_counts = {}
            for year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
                
            # Sort by year
            sorted_years = sorted(year_counts.keys())
            counts = [year_counts[y] for y in sorted_years]
            
            axs[1, 0].bar(sorted_years, counts, color='steelblue')
            axs[1, 0].set_title('Publication Year Distribution')
            axs[1, 0].set_xlabel('Year')
            axs[1, 0].set_ylabel('Number of Papers')
            axs[1, 0].tick_params(axis='x', rotation=45)
        else:
            axs[1, 0].text(0.5, 0.5, "No year data available", 
                          ha='center', va='center')
    else:
        axs[1, 0].text(0.5, 0.5, "No papers analyzed", 
                      ha='center', va='center')
        
    # 4. Weighted verdict visualization
    if summary:
        support_score = summary.get("support_score", 0)
        refute_score = summary.get("refute_score", 0)
        
        verdict_data = {
            "Evidence\nSupporting": support_score,
            "Evidence\nRefuting": refute_score
        }
        
        verdict_colors = ['green', 'red']
        
        axs[1, 1].bar(
            verdict_data.keys(),
            verdict_data.values(),
            color=verdict_colors
        )
        axs[1, 1].set_title('Weighted Evidence Scores')
        axs[1, 1].set_ylabel('Weighted Score')
        axs[1, 1].set_ylim(0, 1)
        
        # Add verdict as text
        validity = summary.get("validity", "inconclusive").capitalize()
        confidence = summary.get("confidence", 0)
        
        verdict_text = f"Verdict: {validity}\nConfidence: {confidence:.2f}"
        axs[1, 1].text(0.5, 0.9, verdict_text, 
                      ha='center', va='top',
                      transform=axs[1, 1].transAxes,
                      bbox=dict(facecolor='white', alpha=0.8))
    else:
        axs[1, 1].text(0.5, 0.5, "No verdict data available", 
                      ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def generate_report(
    claim: str,
    summary: Dict,
    evaluations: List[Dict],
    metadata: List[Dict],
    indices: List[int],
    output_file: Optional[str] = None
) -> str:
    """
    Generate a comprehensive report of the validation results.
    
    Args:
        claim: Scientific claim
        summary: Validation summary
        evaluations: List of paper evaluations
        metadata: List of paper metadata
        indices: List of paper indices
        output_file: Optional file to save the report to
        
    Returns:
        Report text
    """
    if not summary:
        return "No validation results available."
        
    # Create header
    report = f"""# Scientific Claim Validation Report

## Claim Analyzed
"{claim}"

## Validation Verdict
- **Status**: {summary['validity'].capitalize()}
- **Confidence**: {summary['confidence']:.2f}/1.00

## Evidence Summary
- **Total papers retrieved**: {summary['total_papers_retrieved']}
- **Papers analyzed in detail**: {summary['total_papers_analyzed']}
- **Supporting evidence**: {summary['stance_counts']['support']} papers
- **Refuting evidence**: {summary['stance_counts']['refute']} papers
- **Neutral evidence**: {summary['stance_counts']['neutral']} papers

## Scientific Assessment
{summary['llm_synthesis']}

## Evidence Details
"""
    
    # Add key papers
    supporting_papers = []
    refuting_papers = []
    
    for i, (eval_result, idx) in enumerate(zip(evaluations, indices)):
        if "stance" not in eval_result:
            continue
            
        paper_meta = metadata[idx]
        authors = paper_meta.get("authors", ["Unknown"])
        first_author = authors[0] if authors else "Unknown"
        year = paper_meta.get("year", "Unknown")
        title = paper_meta.get("title", "Untitled")
        journal = paper_meta.get("journal", "Unknown journal")
        pmid = paper_meta.get("pmid", "")
        
        paper_entry = {
            "citation": f"{first_author} et al. ({year}). {title}. {journal}.",
            "pmid": pmid,
            "year": year,
            "stance": eval_result["stance"],
            "confidence": eval_result.get("confidence", 0),
            "evidence_quality": eval_result.get("evidence_quality", "Unknown"),
            "relevance": eval_result.get("relevance", "Unknown"),
            "reasoning": eval_result.get("reasoning", "")
        }
        
        if eval_result["stance"] == "support" and eval_result.get("confidence", 0) > 0.6:
            supporting_papers.append(paper_entry)
        elif eval_result["stance"] == "refute" and eval_result.get("confidence", 0) > 0.6:
            refuting_papers.append(paper_entry)
            
    # Sort by confidence
    supporting_papers.sort(key=lambda x: x["confidence"], reverse=True)
    refuting_papers.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Add supporting papers
    if supporting_papers:
        report += "\n### Key Supporting Papers\n"
        for i, paper in enumerate(supporting_papers[:5]):  # Top 5
            report += f"{i+1}. {paper['citation']}\n"
            report += f"   - PMID: {paper['pmid']}\n"
            report += f"   - Confidence: {paper['confidence']:.2f}, Quality: {paper['evidence_quality']}\n"
            report += f"   - Rationale: {paper['reasoning']}\n\n"
    else:
        report += "\n### No Strong Supporting Papers Found\n"
        
    # Add refuting papers
    if refuting_papers:
        report += "\n### Key Refuting Papers\n"
        for i, paper in enumerate(refuting_papers[:5]):  # Top 5
            report += f"{i+1}. {paper['citation']}\n"
            report += f"   - PMID: {paper['pmid']}\n"
            report += f"   - Confidence: {paper['confidence']:.2f}, Quality: {paper['evidence_quality']}\n"
            report += f"   - Rationale: {paper['reasoning']}\n\n"
    else:
        report += "\n### No Strong Refuting Papers Found\n"
        
    # Add methodology
    report += f"""
## Methodology
This assessment was conducted using:
1. Comprehensive PubMed literature retrieval
2. Semantic embedding using science-specific language models
3. Evidence assessment of the most relevant studies
4. Weighted synthesis based on evidence quality and relevance

## Timestamp
Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Save if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
            
    return report