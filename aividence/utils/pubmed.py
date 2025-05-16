"""
PubMed utilities for fetching and parsing literature
"""

import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Tuple
from Bio import Entrez
from tqdm import tqdm

def generate_search_terms(llm_client, llm_model: str, claim: str) -> List[str]:
    """
    Generate comprehensive search terms from a claim using LLM.
    
    Args:
        llm_client: LLM API client
        llm_model: Model to use for generation
        claim: Scientific claim
        
    Returns:
        List of search terms
    """
    prompt = f"""Analyze this scientific claim and generate comprehensive PubMed search terms to validate it.
Don't limit the number of terms - include as many relevant search terms as possible.

CLAIM: {claim}

Guidelines for generating search terms:
1. Focus on the key relationships in the claim (not just isolated keywords)
2. Include the main entity + action combinations (e.g., "EntityX inhibits")
3. Include the main entity + target process (e.g., "EntityX migration")
4. Include any alternative names for biological entities (e.g., "VAP-1" for "AOC3")
5. Include the entity by itself to get general information about it
6. Create different combinations that would capture all relevant literature
7. Be comprehensive - we want to find ALL papers that could validate this claim

Return ONLY the search terms as a simple list, one per line, without numbering or explanation.
"""
    
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        # Parse the response to extract search terms
        search_terms = []
        lines = response.choices[0].message.content.strip().split('\n')
        
        for line in lines:
            term = line.strip()
            if term and not term.startswith('#') and not term.startswith('*'):
                # Remove any leading dashes, numbers or bullets
                term = re.sub(r'^[-•*\d.)\s]+', '', term).strip()
                if term:
                    search_terms.append(term)
        
        # Ensure we have at least some basic search terms if the LLM fails
        if not search_terms:
            raise Exception("No valid search terms generated")
            
    except Exception as e:
        print(f"Error generating search terms with LLM: {e}")
        # Fallback: extract basic terms
        search_terms = []
        
        # Extract capitalized entities (potential genes/proteins)
        capitalized_terms = re.findall(r'\b[A-Z][A-Za-z0-9]*\b', claim)
        
        if capitalized_terms:
            main_entity = capitalized_terms[0]
            search_terms.append(main_entity)
            
            # Add basic context
            if "cancer" in claim.lower():
                search_terms.append(f"{main_entity} cancer")
            if "tumor" in claim.lower():
                search_terms.append(f"{main_entity} tumor")
            if "migration" in claim.lower():
                search_terms.append(f"{main_entity} migration")
            if "progression" in claim.lower():
                search_terms.append(f"{main_entity} progression")
        else:
            # No identifiable entity, use key phrases
            words = claim.split()
            key_terms = [w for w in words if len(w) > 3 and w.lower() not in 
                        ["with", "that", "this", "from", "then", "than", "when"]]
            search_terms = key_terms[:5]
    
    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in search_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    return unique_terms

def parse_pubmed_record(record: str) -> Dict:
    """
    Parse a PubMed record in MEDLINE format.
    
    Args:
        record: MEDLINE record text
        
    Returns:
        Dictionary with extracted data
    """
    result = {
        'pmid': None,
        'title': None,
        'abstract': None,
        'authors': [],
        'year': None,
        'journal': None,
        'doi': None
    }
    
    if not record.strip():
        return None
        
    lines = record.split("\n")
    
    # Extract data fields
    for i, line in enumerate(lines):
        if line.startswith("PMID- "):
            result['pmid'] = line[6:].strip()
        elif line.startswith("TI  - "):
            # Title might span multiple lines
            result['title'] = line[6:].strip()
            j = i + 1
            while j < len(lines) and lines[j].startswith("      "):
                result['title'] += " " + lines[j].strip()
                j += 1
        elif line.startswith("AB  - "):
            # Abstract might span multiple lines
            result['abstract'] = line[6:].strip()
            j = i + 1
            while j < len(lines) and lines[j].startswith("      "):
                result['abstract'] += " " + lines[j].strip()
                j += 1
        elif line.startswith("AU  - "):
            result['authors'].append(line[6:].strip())
        elif line.startswith("DP  - "):
            # Extract year from date
            date_parts = line[6:].strip().split()
            if date_parts:
                try:
                    result['year'] = int(date_parts[0][:4])
                except:
                    pass
        elif line.startswith("JT  - "):
            result['journal'] = line[6:].strip()
        elif line.startswith("LID  - "):
            # Try to extract DOI
            lid = line[6:].strip()
            if "[doi]" in lid:
                result['doi'] = lid.split("[doi]")[0].strip()
    
    return result

def search_pubmed_term(term: str, email: str, cache_dir: str = None, max_results: int = 200) -> Tuple[List, List]:
    """
    Search PubMed for a single term.
    
    Args:
        term: Search term
        email: Email for PubMed API
        cache_dir: Optional cache directory
        max_results: Maximum results to return
        
    Returns:
        Tuple of (abstracts, metadata)
    """
    print(f"Searching for: {term}")
    term_abstracts = []
    term_metadata = []
    
    # Check cache first
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"search_{hash(term) % 10000}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            print(f"Loaded {len(cache_data['abstracts'])} results from cache for '{term}'")
            return cache_data['abstracts'], cache_data['metadata']
    
    # Search PubMed
    try:
        Entrez.email = email
        handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
        record = Entrez.read(handle)
        id_list = record["IdList"]
        
        if id_list:
            print(f"  Found {len(id_list)} results")
            
            # Fetch in batches to avoid timeouts
            batch_size = 50
            
            # 添加批量处理的进度条
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i+batch_size]
                
                try:
                    handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
                    records = handle.read()
                    
                    # Process records
                    record_list = records.split("\n\n\n")
                    for record in record_list:
                        if not record.strip():
                            continue
                            
                        # Parse the record
                        paper_data = parse_pubmed_record(record)
                        
                        if paper_data and paper_data['abstract']:
                            term_abstracts.append(paper_data['abstract'])
                            
                            # Create metadata entry
                            metadata_entry = {k: v for k, v in paper_data.items() if k != 'abstract'}
                            term_metadata.append(metadata_entry)
                            
                except Exception as e:
                    print(f"  Error fetching batch: {e}")
            
            # Cache the results
            if cache_dir:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                    
                cache_data = {
                    'abstracts': term_abstracts,
                    'metadata': term_metadata
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
        else:
            print("  No results found")
                
    except Exception as e:
        print(f"Error searching for '{term}': {e}")
    
    time.sleep(0.5)  # Be nice to the API
    return term_abstracts, term_metadata

def fetch_literature_parallel(search_terms: List[str], email: str, cache_dir: str = None, 
                             max_results: int = 200, max_workers: int = 5) -> Tuple[List, List]:
    """
    Fetch scientific literature based on search terms in parallel.
    
    Args:
        search_terms: List of search terms
        email: Email for PubMed API
        cache_dir: Optional cache directory
        max_results: Maximum results per term
        max_workers: Maximum number of parallel workers
        
    Returns:
        Tuple of (abstracts, metadata)
    """
    all_abstracts = []
    all_metadata = []
    
    print(f"Searching PubMed with {len(search_terms)} terms using {max_workers} parallel workers...")
    
    # Use ThreadPoolExecutor for parallel searches with tqdm progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建任务
        future_to_term = {
            executor.submit(search_pubmed_term, term, email, cache_dir, max_results): term 
            for term in search_terms
        }
        
        # 用tqdm添加进度条
        for future in tqdm(as_completed(future_to_term), total=len(search_terms), 
                          desc="Searching PubMed", unit="term"):
            term = future_to_term[future]
            try:
                term_abstracts, term_metadata = future.result()
                all_abstracts.extend(term_abstracts)
                all_metadata.extend(term_metadata)
            except Exception as e:
                print(f"Error processing term '{term}': {e}")
    
    print("Removing duplicate papers...")
    
    # Remove duplicates by PMID with progress bar
    unique_abstracts = []
    unique_metadata = []
    seen_pmids = set()
    
    # 使用tqdm为去重过程添加进度条
    for abstract, metadata in tqdm(zip(all_abstracts, all_metadata), 
                                  total=len(all_abstracts),
                                  desc="Deduplicating papers", unit="paper"):
        pmid = metadata.get('pmid')
        if pmid and pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique_abstracts.append(abstract)
            unique_metadata.append(metadata)
            
    print(f"Found {len(unique_abstracts)} unique papers after deduplication")
    
    return unique_abstracts, unique_metadata