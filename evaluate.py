#!/usr/bin/env python3
"""
Risk Comparison Prototype using DeepInfra API

This script compares risks from MIT Risk Repository against identified risks in Komdigi
to determine if the risks from MIT Risk Repository is covered by risks in Komdigi.

The comparison is done using LLMs

Requirements:
- pip install openai pandas
- Set DEEPINFRA_API_TOKEN environment variable
"""

import os
import json
import pandas as pd
from openai import OpenAI
import time
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskComparator:
    """
    Compare risks between Document A and Document B using DeepInfra LLM API
    """
    
    def __init__(self, api_token: str = None, model_name: str = "deepseek-ai/DeepSeek-V3"):
        """
        Initialize the RiskComparator
        
        Args:
            api_token: DeepInfra API token (if None, reads from environment)
            model_name: Model to use for comparison
        """
        self.api_token = api_token or os.getenv("DEEPINFRA_API_TOKEN")
        if not self.api_token:
            raise ValueError("DeepInfra API token is required. Set DEEPINFRA_API_TOKEN environment variable.")
        
        self.model_name = model_name
        self.client = OpenAI(
            api_key=self.api_token,
            base_url="https://api.deepinfra.com/v1/openai"
        )
        
        # Inference parameters for consistency
        self.inference_params = {
            "temperature": 0.0,  # Maximum consistency
            "top_p": 1.0,       # Top-1 sampling equivalent
            "max_tokens": 500,  # Sufficient for JSON response
            "stop": None
        }
    
    def load_mit_risks(self, file_path: str, limit: int = 100) -> pd.DataFrame:
        """
        Load Document A (comprehensive risk list) and limit to first N risks
        
        Expected columns: Title,QuickRef,Ev_ID,Paper_ID,Cat_ID,SubCat_ID,AddEv_ID,
        Category level,Risk category,Risk subcategory,Description,Additional ev.,
        P.Def,p.AddEv,Entity,Intent,Timing,Domain,Sub-domain
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded Document A with {len(df)} total risks")
            
            # Limit to first N risks for token usage measurement
            df_limited = df.head(limit)
            logger.info(f"Limited to first {len(df_limited)} risks for processing")
            
            return df_limited
        except Exception as e:
            logger.error(f"Error loading Document A: {e}")
            raise
    
    def load_komdigi_risks(self, file_path: str) -> pd.DataFrame:
        """
        Load Komdigi (identified risks)
        
        Expected columns: Risk ID,Page,Risk Type (English),Description
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded Komdigi document with {len(df)} identified risks")
            return df
        except Exception as e:
            logger.error(f"Error loading Komdigi: {e}")
            raise
    
    def create_prompt(self, risk_a: Dict[str, Any], risks_b: pd.DataFrame) -> str:
        """
        Create a prompt for the LLM to compare one risk from A against all risks in B
        """
        # Extract key information from risk A

        risk_a_summary = {
            "title": risk_a.get("Title", ""),
            "category": risk_a.get("Risk category", ""),
            "subcategory": risk_a.get("Risk subcategory", ""),
            "description": risk_a.get("Description", ""),
            "domain": risk_a.get("Domain", ""),
            "subdomain": risk_a.get("Sub-domain", ""),
            "additional" : risk_a.get("Additional ev.", "")
        }

        # Format risks from B
        risks_b_formatted = []
        for _, risk_b in risks_b.iterrows():
            risks_b_formatted.append({
                "risk_id": risk_b.get("Risk ID", ""),
                "risk_type": risk_b.get("Risk Type (English)", ""),
                "description": risk_b.get("Description", "")
            })
        
        prompt = f"""You are a risk analysis expert. Your task is to determine if a specific risk from Document A is covered by any of the identified risks in Document B.

**Risk from Document A to analyze:**
- Title: {risk_a_summary['title']}
- Category: {risk_a_summary['category']}
- Subcategory: {risk_a_summary['subcategory']}
- Description: {risk_a_summary['description']}
- Domain: {risk_a_summary['domain']}
- Sub-domain: {risk_a_summary['subdomain']}

**Identified risks from Document B:**
{json.dumps(risks_b_formatted, indent=2)}

**Instructions:**
1. Carefully analyze the risk from Document A
2. Compare it against ALL risks in Document B
3. Determine if any risk in Document B covers, addresses, or is equivalent to the risk from Document A
4. Consider semantic similarity, not just exact word matching
5. A risk is "covered" if Document B identifies the same or substantially similar risk concern

**Response Format:**
Respond ONLY with a valid JSON object in exactly this format:
{{
    "identified": true,
    "reason": "Risk A is covered by Risk ID X in Document B because both address the same fundamental concern about [specific explanation]. The descriptions show substantial overlap in [detailed reasoning]."
}}

OR

{{
    "identified": false,
    "reason": "Risk A is not covered by any risk in Document B. While some risks are related, none specifically address [specific aspects]. The closest match is Risk ID X, but it differs because [detailed explanation of differences]."
}}

Ensure your reasoning is explicit, detailed, and clearly explains your decision."""

        return prompt
    
    def query_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Query the LLM with the given prompt and return parsed JSON response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise risk analysis expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                **self.inference_params
            )
            
            response_text = response.choices[0].message.content.strip()

            print(response)
            
            # Log token usage
            usage = response.usage
            logger.info(f"Token usage - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
            # Parse JSON response
            try:
                result = response_text.replace("```json", "")
                result = result[:-3]
                result = json.loads(result)
                if "identified" not in result or "reason" not in result:
                    raise ValueError("Response missing required fields")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {result}")
                # Return a default response if JSON parsing fails
                return {
                    "identified": False,
                    "reason": f"Error parsing LLM response: {str(e)}"
                }
        
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return {
                "identified": False,
                "reason": f"Error querying LLM: {str(e)}"
            }
    
    def compare_risks(self, mit_doc_path: str, komdigi_doc_path: str, risks_limit: int = 100, delay_between_requests: float = 1.0) -> List[Dict[str, Any]]:
        """
        Compare risks from Document A against Document B
        
        Args:
            mit_doc_path: Path to Document A CSV
            komdigi_doc_path: Path to Document B CSV
            limit_a: Limit processing to first N risks from Document A
            delay_between_requests: Delay between API requests to avoid rate limiting
            
        Returns:
            List of comparison results
        """
        # Load documents
        df_a = self.load_mit_risks(mit_doc_path, risks_limit)
        df_b = self.load_komdigi_risks(komdigi_doc_path)
        
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        logger.info(f"Starting comparison of {len(df_a)} risks from MIT against {len(df_b)} risks from Komdigi")
        
        for idx, (_, risk_a) in enumerate(df_a.iterrows(), 1):
            logger.info(f"Processing risk {idx}/{len(df_a)}: {risk_a.get('Title', 'Unknown')}")
            
            # Create prompt
            prompt = self.create_prompt(risk_a.to_dict(), df_b)

            ev_id = risk_a.to_dict()["Ev_ID"]
            
            # Query LLM
            llm_result = self.query_llm(prompt)

            # Store result with metadata
            result = {
                "ev_id" : ev_id,
                "document_a_index": idx - 1,
                "document_a_title": risk_a.get("Title", ""),
                "document_a_category": risk_a.get("Risk category", ""),
                "document_a_description": risk_a.get("Description", ""),
                "identified": llm_result["identified"],
                "reason": llm_result["reason"],
                "model_used": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            results.append(result)
            
            # Add delay between requests
            if delay_between_requests > 0 and idx < len(df_a):
                time.sleep(delay_between_requests)
        
        # Summary statistics
        identified_count = sum(1 for r in results if r["identified"])
        logger.info(f"Comparison complete. {identified_count}/{len(results)} risks were identified as covered.")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to CSV file"""
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of the comparison results"""
        total = len(results)
        identified = sum(1 for r in results if r["identified"])
        not_identified = total - identified
        
        print(f"\n{'='*60}")
        print("RISK COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Total risks analyzed: {total}")
        print(f"Risks identified as covered: {identified} ({identified/total*100:.1f}%)")
        print(f"Risks not covered: {not_identified} ({not_identified/total*100:.1f}%)")
        print(f"Model used: {self.model_name}")
        print(f"{'='*60}")
        
        # Show first few examples of each type
        print("\nEXAMPLE COVERED RISKS:")
        covered_examples = [r for r in results if r["identified"]][:3]
        for i, result in enumerate(covered_examples, 1):
            print(f"\n{i}. {result['document_a_title']}")
            print(f"   Reason: {result['reason'][:200]}...")
        
        print("\nEXAMPLE UNCOVERED RISKS:")
        uncovered_examples = [r for r in results if not r["identified"]][:3]
        for i, result in enumerate(uncovered_examples, 1):
            print(f"\n{i}. {result['document_a_title']}")
            print(f"   Reason: {result['reason'][:200]}...")


def main():
    """
    Example usage of the RiskComparator
    """
    # Initialize comparator
    # You can specify a different model, e.g.:
    # - "deepseek-ai/DeepSeek-V3" (default, good balance of cost/performance)
    # - "zai-org/GLM-4.5" (alternative option)
    # - "meta-llama/Meta-Llama-3.1-70B-Instruct" (higher cost but potentially better)
    
    comparator = RiskComparator(
        model_name="microsoft/phi-4",  # Cost-effective choice
        api_token= # Insert token here
    )
    
    # File paths (update these with your actual file paths)
    mit_doc_path = "data/AI Risk Repository V3_26_03_2025 - AI Risk Database v3.csv"  # Your comprehensive risk document
    komdigi_doc_path = "data/Identified Risks from Peta Jalan KA.csv"  # Your identified risks document
    output_path = "risk_comparison_results.csv"
    
    try:
        # Run comparison on first 100 risks from Document A
        results = comparator.compare_risks(
            mit_doc_path=mit_doc_path,
            komdigi_doc_path=komdigi_doc_path,
            risks_limit=9999,  # First 100 risks for token measurement
            delay_between_requests=1.0  # 1-second delay between requests
        )
        
        # Save results
        comparator.save_results(results, output_path)
        
        # Print summary
        comparator.print_summary(results)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()