#!/usr/bin/env python3
"""
Accuracy Testing Script for HackRx Smart RAG System
This script helps evaluate the accuracy of your RAG system answers.
"""

import requests
import json
import time
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
# Also works with ngrok URL: "https://8ce83bbba509.ngrok-free.app"
AUTH_TOKEN = "09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b"

# Enhanced test cases with more comprehensive coverage
TEST_CASES = [
    # Coverage Questions
    {
        "question": "What is the deductible for medical coverage?",
        "expected_keywords": ["deductible", "medical", "$", "coverage", "amount"],
        "expected_type": "amount",
        "category": "coverage",
        "difficulty": "easy",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "Does the policy cover dental treatment?",
        "expected_keywords": ["yes", "no", "dental", "covered", "treatment"],
        "expected_type": "yes_no",
        "category": "coverage",
        "difficulty": "easy",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "What are the exclusions in this policy?",
        "expected_keywords": ["exclusions", "not covered", "excluded", "limitations"],
        "expected_type": "descriptive",
        "category": "exclusions",
        "difficulty": "medium",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    
    # Financial Questions
    {
        "question": "What is the maximum coverage limit?",
        "expected_keywords": ["maximum", "limit", "coverage", "$", "amount"],
        "expected_type": "amount",
        "category": "financial",
        "difficulty": "easy",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "How much is the annual premium?",
        "expected_keywords": ["premium", "annual", "cost", "$", "payment"],
        "expected_type": "amount",
        "category": "financial",
        "difficulty": "easy",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    
    # Procedure Questions
    {
        "question": "How do I file a claim?",
        "expected_keywords": ["claim", "file", "procedure", "process", "submit"],
        "expected_type": "descriptive",
        "category": "procedures",
        "difficulty": "medium",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "What documents are required for claim processing?",
        "expected_keywords": ["documents", "required", "claim", "processing", "submit"],
        "expected_type": "descriptive",
        "category": "procedures",
        "difficulty": "medium",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    
    # Complex/Difficult Questions
    {
        "question": "What is the waiting period for pre-existing conditions?",
        "expected_keywords": ["waiting", "period", "pre-existing", "conditions", "months", "years"],
        "expected_type": "descriptive",
        "category": "complex",
        "difficulty": "hard",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "Are there any age-specific coverage limitations?",
        "expected_keywords": ["age", "limitations", "coverage", "specific", "restrictions"],
        "expected_type": "descriptive",
        "category": "complex",
        "difficulty": "hard",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    
    # Edge Cases
    {
        "question": "Does this policy cover emergency ambulance services?",
        "expected_keywords": ["emergency", "ambulance", "services", "covered", "yes", "no"],
        "expected_type": "yes_no",
        "category": "edge_case",
        "difficulty": "medium",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "What happens if I miss a premium payment?",
        "expected_keywords": ["miss", "premium", "payment", "grace", "period", "lapse"],
        "expected_type": "descriptive",
        "category": "edge_case",
        "difficulty": "hard",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    
    # Specific Coverage Items
    {
        "question": "Is hospitalization for COVID-19 covered?",
        "expected_keywords": ["covid", "coronavirus", "hospitalization", "covered", "pandemic"],
        "expected_type": "yes_no",
        "category": "specific",
        "difficulty": "medium",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    },
    {
        "question": "Does the policy cover maternity benefits?",
        "expected_keywords": ["maternity", "benefits", "pregnancy", "covered", "childbirth"],
        "expected_type": "yes_no",
        "category": "specific",
        "difficulty": "easy",
        "document_url": "https://raw.githubusercontent.com/redditard/HackRx-Tests/main/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf"
    }
]

class AccuracyTester:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        self.results = []
    
    def test_single_question(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single question and evaluate the answer with enhanced metrics."""
        try:
            # Make API request
            payload = {
                "documents": test_case["document_url"],
                "questions": [test_case["question"]]
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=90  # Increased timeout for enhanced processing
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                answer_data = response_data["answers"][0]
                
                # Extract answer text and additional metrics
                if isinstance(answer_data, dict):
                    answer_text = answer_data.get("answer", str(answer_data))
                    confidence = answer_data.get("confidence", 0.0)
                    validation = answer_data.get("validation", {})
                    sources = answer_data.get("sources", [])
                    search_method = answer_data.get("search_method", "unknown")
                else:
                    answer_text = str(answer_data)
                    confidence = 0.0
                    validation = {}
                    sources = []
                    search_method = "unknown"
                
                # Evaluate answer with enhanced scoring
                evaluation = self.evaluate_answer_enhanced(answer_text, test_case)
                
                result = {
                    "question": test_case["question"],
                    "category": test_case.get("category", "unknown"),
                    "difficulty": test_case.get("difficulty", "unknown"),
                    "answer": answer_text,
                    "expected_type": test_case["expected_type"],
                    "expected_keywords": test_case["expected_keywords"],
                    
                    # Enhanced metrics
                    "accuracy_score": evaluation["accuracy_score"],
                    "keyword_match_score": evaluation["keyword_score"],
                    "type_match_score": evaluation["type_score"],
                    "completeness_score": evaluation["completeness_score"],
                    "clarity_score": evaluation["clarity_score"],
                    
                    # System metrics
                    "system_confidence": confidence,
                    "validation_passed": validation.get("is_valid", False),
                    "validation_score": validation.get("confidence_score", 0.0),
                    "factual_accuracy": validation.get("factual_accuracy", 0.0),
                    "search_method": search_method,
                    "source_count": len(sources),
                    
                    "response_time": response_time,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "issues": validation.get("issues", []),
                    "suggestions": validation.get("suggestions", [])
                }
            else:
                result = {
                    "question": test_case["question"],
                    "answer": f"Error: {response.status_code} - {response.text}",
                    "accuracy_score": 0.0,
                    "response_time": response_time,
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            result = {
                "question": test_case["question"],
                "answer": f"Exception: {str(e)}",
                "accuracy_score": 0.0,
                "response_time": 0.0,
                "status": "exception",
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    def evaluate_answer_enhanced(self, answer: str, test_case: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced answer evaluation with multiple scoring dimensions."""
        answer_lower = answer.lower()
        
        # 1. Keyword matching score (25%)
        keyword_matches = 0
        for keyword in test_case["expected_keywords"]:
            if keyword.lower() in answer_lower:
                keyword_matches += 1
        keyword_score = keyword_matches / len(test_case["expected_keywords"]) if test_case["expected_keywords"] else 0.0
        
        # 2. Answer type appropriateness (25%)
        type_score = self._evaluate_answer_type(answer, test_case["expected_type"])
        
        # 3. Completeness score (25%)
        completeness_score = self._evaluate_completeness(answer, test_case)
        
        # 4. Clarity and specificity score (25%)
        clarity_score = self._evaluate_clarity(answer)
        
        # Overall accuracy score
        accuracy_score = (keyword_score * 0.25 + type_score * 0.25 + 
                         completeness_score * 0.25 + clarity_score * 0.25)
        
        return {
            "accuracy_score": accuracy_score,
            "keyword_score": keyword_score,
            "type_score": type_score,
            "completeness_score": completeness_score,
            "clarity_score": clarity_score
        }
    
    def _evaluate_answer_type(self, answer: str, expected_type: str) -> float:
        """Evaluate if answer matches expected type."""
        answer_lower = answer.lower()
        
        if expected_type == "amount":
            # Look for monetary amounts, percentages, or numbers
            import re
            if re.search(r'\$[\d,]+|\d+%|\d+ (dollars?|rupees?|percent)', answer):
                return 1.0
            elif re.search(r'\d+', answer):
                return 0.7  # Has numbers but not clearly formatted as amount
            else:
                return 0.0
                
        elif expected_type == "yes_no":
            if any(word in answer_lower for word in ["yes", "no", "covered", "not covered", "included", "excluded"]):
                return 1.0
            else:
                return 0.3  # Partial credit for descriptive answers
                
        elif expected_type == "descriptive":
            if len(answer.strip()) > 50:  # Substantial descriptive answer
                return 1.0
            elif len(answer.strip()) > 20:
                return 0.7
            else:
                return 0.3
        
        return 0.5  # Default partial credit
    
    def _evaluate_completeness(self, answer: str, test_case: Dict[str, Any]) -> float:
        """Evaluate answer completeness based on question complexity."""
        answer_length = len(answer.strip())
        difficulty = test_case.get("difficulty", "medium")
        
        # Minimum expected lengths based on difficulty
        min_lengths = {"easy": 30, "medium": 60, "hard": 100}
        good_lengths = {"easy": 80, "medium": 150, "hard": 250}
        
        min_len = min_lengths.get(difficulty, 60)
        good_len = good_lengths.get(difficulty, 150)
        
        if answer_length >= good_len:
            return 1.0
        elif answer_length >= min_len:
            return 0.7
        elif answer_length >= min_len * 0.5:
            return 0.4
        else:
            return 0.1
    
    def _evaluate_clarity(self, answer: str) -> float:
        """Evaluate answer clarity and specificity."""
        score = 0.0
        answer_lower = answer.lower()
        
        # Positive indicators
        if '"' in answer or "according to" in answer_lower or "policy states" in answer_lower:
            score += 0.3  # References policy directly
        
        if any(word in answer_lower for word in ["specific", "exactly", "precisely", "states that"]):
            score += 0.2  # Uses specific language
        
        if not any(word in answer_lower for word in ["might", "could", "possibly", "perhaps", "maybe"]):
            score += 0.3  # Avoids uncertain language
        
        # Check for structure (lists, steps, etc.)
        if any(pattern in answer for pattern in ["1.", "2.", "•", "-", "step", "first", "second"]):
            score += 0.2  # Well-structured
        
        return min(score, 1.0)
    
    def evaluate_answer(self, answer: str, test_case: Dict[str, Any]) -> float:
        """Legacy method for backward compatibility."""
        evaluation = self.evaluate_answer_enhanced(answer, test_case)
        return evaluation["accuracy_score"]
    
    def run_all_tests(self, test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """Run all test cases and return enhanced results."""
        print("🧪 Starting enhanced accuracy testing...")
        print(f"📊 Running {len(test_cases)} test cases across multiple categories\n")
        
        for i, test_case in enumerate(test_cases, 1):
            category = test_case.get('category', 'unknown')
            difficulty = test_case.get('difficulty', 'unknown')
            print(f"Testing {i}/{len(test_cases)} [{category}/{difficulty}]: {test_case['question'][:60]}...")
            
            result = self.test_single_question(test_case)
            self.results.append(result)
            
            # Print immediate feedback
            if result['status'] == 'success':
                accuracy = result.get('accuracy_score', 0.0)
                validation_passed = result.get('validation_passed', False)
                search_method = result.get('search_method', 'unknown')
                
                print(f"   ✅ Accuracy: {accuracy:.2f} | Validation: {'✓' if validation_passed else '✗'} | Method: {search_method}")
                
                if result.get('issues'):
                    print(f"   ⚠️  Issues: {', '.join(result['issues'][:2])}")
            else:
                print(f"   ❌ {result['status']}: {result.get('answer', 'Unknown error')[:50]}")
            
            print()  # Empty line for readability
        
        return pd.DataFrame(self.results)
    
    def generate_enhanced_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive enhanced accuracy report."""
        successful_tests = df[df['status'] == 'success']
        
        if len(successful_tests) == 0:
            return "❌ No successful tests to analyze!"
        
        # Overall statistics
        avg_accuracy = successful_tests['accuracy_score'].mean()
        avg_response_time = successful_tests['response_time'].mean()
        success_rate = len(successful_tests) / len(df) * 100
        
        # Enhanced metrics
        avg_validation_score = successful_tests['validation_score'].mean()
        validation_pass_rate = successful_tests['validation_passed'].sum() / len(successful_tests) * 100
        avg_factual_accuracy = successful_tests['factual_accuracy'].mean()
        
        # Category analysis
        category_stats = successful_tests.groupby('category').agg({
            'accuracy_score': ['mean', 'count'],
            'validation_passed': 'sum'
        }).round(2)
        
        # Difficulty analysis
        difficulty_stats = successful_tests.groupby('difficulty').agg({
            'accuracy_score': ['mean', 'count'],
            'response_time': 'mean'
        }).round(2)
        
        # Search method analysis
        search_method_stats = successful_tests.groupby('search_method').agg({
            'accuracy_score': ['mean', 'count']
        }).round(2)
        
        report = f"""
🎯 ENHANCED ACCURACY REPORT
{'='*60}
📊 Overall Statistics:
   • Success Rate: {success_rate:.1f}% ({len(successful_tests)}/{len(df)} tests)
   • Average Accuracy Score: {avg_accuracy:.2f}/1.0 ({avg_accuracy*100:.1f}%)
   • Average Response Time: {avg_response_time:.1f} seconds
   • Validation Pass Rate: {validation_pass_rate:.1f}%
   • Average Validation Score: {avg_validation_score:.2f}/1.0
   • Average Factual Accuracy: {avg_factual_accuracy:.2f}/1.0

📈 Score Distribution:
   • Excellent (>0.8): {len(successful_tests[successful_tests['accuracy_score'] > 0.8])} tests
   • Good (0.6-0.8): {len(successful_tests[(successful_tests['accuracy_score'] >= 0.6) & (successful_tests['accuracy_score'] <= 0.8)])} tests
   • Poor (<0.6): {len(successful_tests[successful_tests['accuracy_score'] < 0.6])} tests

📋 Category Performance:
{category_stats.to_string()}

🎚️ Difficulty Analysis:
{difficulty_stats.to_string()}

🔍 Search Method Performance:
{search_method_stats.to_string()}

⚡ Performance Insights:
   • Best performing category: {successful_tests.groupby('category')['accuracy_score'].mean().idxmax()}
   • Most challenging difficulty: {successful_tests.groupby('difficulty')['accuracy_score'].mean().idxmin()}
   • Optimal search method: {successful_tests.groupby('search_method')['accuracy_score'].mean().idxmax()}

🚨 Common Issues:
"""
        
        # Add common issues analysis
        all_issues = []
        for issues in successful_tests['issues']:
            if isinstance(issues, list):
                all_issues.extend(issues)
        
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            for issue, count in issue_counts.most_common(5):
                report += f"   • {issue}: {count} occurrences\n"
        
        return report

⚡ Performance:
   • Fastest Response: {successful_tests['response_time'].min():.1f}s
   • Slowest Response: {successful_tests['response_time'].max():.1f}s
"""
        
        # Add detailed results
        report += "\n📋 DETAILED RESULTS:\n" + "="*50 + "\n"
        for _, row in df.iterrows():
            status_emoji = "✅" if row['status'] == 'success' else "❌"
            report += f"{status_emoji} Score: {row['accuracy_score']:.2f} | {row['response_time']:.1f}s\n"
            report += f"❓ Q: {row['question']}\n"
            report += f"💬 A: {row['answer'][:150]}...\n"
            report += "-" * 30 + "\n"
        
        return report

def main():
    """Main function to run accuracy tests."""
    print("🚀 HackRx Smart RAG - Accuracy Testing")
    print("="*50)
    
    # Initialize tester
    tester = AccuracyTester(API_BASE_URL, AUTH_TOKEN)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running and healthy")
        else:
            print(f"⚠️ Server health check failed: {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure your server is running on http://localhost:8000")
        return
    
    # Run tests
    results_df = tester.run_all_tests(TEST_CASES)
    
    # Generate and display report
    report = tester.generate_report(results_df)
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"accuracy_test_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"💾 Results saved to: {csv_filename}")
    
    # Save report
    report_filename = f"accuracy_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📄 Report saved to: {report_filename}")

if __name__ == "__main__":
    main()
