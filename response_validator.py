"""
Response Validation System
High Impact, Low Effort Optimization for improving answer accuracy
"""

import re
import asyncio
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of response validation"""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    suggestions: List[str]
    factual_accuracy: float
    completeness_score: float

class ResponseValidator:
    """Validates LLM responses for accuracy and completeness"""
    
    def __init__(self):
        self.policy_terms = {
            'coverage_terms': ['coverage', 'covered', 'benefit', 'protection', 'included'],
            'exclusion_terms': ['excluded', 'not covered', 'exception', 'limitation', 'restriction'],
            'financial_terms': ['premium', 'deductible', 'copay', 'coinsurance', 'out-of-pocket', 'maximum', 'limit'],
            'time_terms': ['annually', 'monthly', 'days', 'period', 'effective', 'renewal'],
            'claim_terms': ['claim', 'procedure', 'process', 'file', 'submit', 'reimbursement']
        }
        
        # Common hallucination patterns
        self.hallucination_indicators = [
            r'typically', r'usually', r'generally', r'most policies',
            r'in most cases', r'often', r'commonly', r'standard practice'
        ]
        
        # Required components for different question types
        self.response_requirements = {
            'coverage': ['specific items covered', 'conditions or limitations'],
            'exclusions': ['specific exclusions', 'exact policy language'],
            'amounts': ['specific amounts', 'currencies or percentages'],
            'procedures': ['step-by-step process', 'time requirements'],
            'general': ['direct answer', 'policy reference']
        }
    
    async def validate_response(self, question: str, response: str, context: str) -> ValidationResult:
        """Comprehensive response validation"""
        
        issues = []
        suggestions = []
        
        # 1. Basic validation checks
        basic_checks = self._perform_basic_validation(response)
        issues.extend(basic_checks['issues'])
        suggestions.extend(basic_checks['suggestions'])
        
        # 2. Factual accuracy against context
        factual_score = self._check_factual_accuracy(response, context)
        if factual_score < 0.7:
            issues.append("Response may contain information not supported by the provided context")
            suggestions.append("Ensure all facts come directly from the policy document")
        
        # 3. Completeness check
        completeness_score = self._check_completeness(question, response)
        if completeness_score < 0.6:
            issues.append("Response may be incomplete for the type of question asked")
            suggestions.append("Consider addressing all aspects of the question")
        
        # 4. Hallucination detection
        hallucination_score = self._detect_hallucinations(response)
        if hallucination_score > 0.3:
            issues.append("Response contains language that suggests general knowledge rather than specific policy information")
            suggestions.append("Use exact policy language and avoid generalizations")
        
        # 5. Policy terminology consistency
        terminology_score = self._check_terminology_consistency(response, context)
        if terminology_score < 0.8:
            issues.append("Response terminology doesn't match the policy document")
            suggestions.append("Use exact terms from the policy document")
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(
            factual_score, completeness_score, 
            1.0 - hallucination_score, terminology_score
        )
        
        is_valid = (len(issues) == 0 and confidence_score > 0.7)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            suggestions=suggestions,
            factual_accuracy=factual_score,
            completeness_score=completeness_score
        )
    
    def _perform_basic_validation(self, response: str) -> Dict[str, List[str]]:
        """Basic validation checks"""
        issues = []
        suggestions = []
        
        # Check response length
        if len(response.strip()) < 20:
            issues.append("Response is too short")
            suggestions.append("Provide more detailed information")
        elif len(response.strip()) > 1000:
            issues.append("Response may be too verbose")
            suggestions.append("Focus on the most relevant information")
        
        # Check for non-committal language
        non_committal = ['might', 'could be', 'possibly', 'perhaps', 'maybe']
        if any(phrase in response.lower() for phrase in non_committal):
            issues.append("Response contains uncertain language")
            suggestions.append("Provide definitive answers based on policy language")
        
        # Check for placeholder responses
        placeholders = ['[amount]', '[specific]', '[details]', 'information not available']
        if any(placeholder in response.lower() for placeholder in placeholders):
            if 'information not available' not in response.lower():
                issues.append("Response contains placeholder text")
                suggestions.append("Replace placeholders with actual policy information")
        
        return {'issues': issues, 'suggestions': suggestions}
    
    def _check_factual_accuracy(self, response: str, context: str) -> float:
        """Check if response facts are supported by context"""
        if not context or not response:
            return 0.0
        
        # Extract key facts from response (numbers, percentages, specific terms)
        response_facts = self._extract_facts(response)
        context_facts = self._extract_facts(context)
        
        if not response_facts:
            return 0.8  # No specific facts to verify
        
        supported_facts = 0
        for fact in response_facts:
            if any(self._facts_match(fact, context_fact) for context_fact in context_facts):
                supported_facts += 1
        
        return supported_facts / len(response_facts) if response_facts else 0.0
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract verifiable facts from text"""
        facts = []
        
        # Extract numbers and percentages
        numbers = re.findall(r'\$?[\d,]+\.?\d*%?', text)
        facts.extend(numbers)
        
        # Extract dates
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d+ (days?|months?|years?)', text)
        facts.extend(dates)
        
        # Extract specific policy terms (capitalized phrases)
        policy_phrases = re.findall(r'[A-Z][A-Za-z\s]{2,}(?=[.,:;]|\s+[a-z])', text)
        facts.extend([phrase.strip() for phrase in policy_phrases if len(phrase.strip()) > 3])
        
        return facts
    
    def _facts_match(self, fact1: str, fact2: str) -> bool:
        """Check if two facts match (allowing for minor variations)"""
        # Normalize whitespace and case
        fact1 = re.sub(r'\s+', ' ', fact1.strip().lower())
        fact2 = re.sub(r'\s+', ' ', fact2.strip().lower())
        
        # Exact match
        if fact1 == fact2:
            return True
        
        # Partial match for longer phrases
        if len(fact1) > 10 and len(fact2) > 10:
            return fact1 in fact2 or fact2 in fact1
        
        # Number/percentage matching with tolerance
        if re.match(r'[\d.,]+%?', fact1) and re.match(r'[\d.,]+%?', fact2):
            return fact1 == fact2
        
        return False
    
    def _check_completeness(self, question: str, response: str) -> float:
        """Check if response completely addresses the question"""
        question_type = self._classify_question_type(question)
        requirements = self.response_requirements.get(question_type, self.response_requirements['general'])
        
        score = 0.0
        for requirement in requirements:
            if self._response_meets_requirement(response, requirement):
                score += 1.0 / len(requirements)
        
        # Bonus for addressing question words
        question_words = re.findall(r'\b(what|when|where|why|how|who|which)\b', question.lower())
        if question_words:
            for word in question_words:
                if self._addresses_question_word(word, response):
                    score += 0.1
        
        return min(score, 1.0)
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type based on content"""
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['cover', 'include', 'benefit']):
            return 'coverage'
        elif any(term in question_lower for term in ['exclude', 'not cover', 'exception']):
            return 'exclusions'
        elif any(term in question_lower for term in ['cost', 'premium', 'amount', 'deductible']):
            return 'amounts'
        elif any(term in question_lower for term in ['how', 'process', 'procedure', 'file']):
            return 'procedures'
        else:
            return 'general'
    
    def _response_meets_requirement(self, response: str, requirement: str) -> bool:
        """Check if response meets a specific requirement"""
        response_lower = response.lower()
        
        if requirement == 'specific items covered':
            return bool(re.search(r'(includes?|covers?|benefits?)\s+[^.]{10,}', response_lower))
        elif requirement == 'conditions or limitations':
            return any(term in response_lower for term in ['condition', 'limitation', 'requirement', 'must', 'only if'])
        elif requirement == 'specific exclusions':
            return any(term in response_lower for term in ['excludes?', 'not covered', 'except', 'limitation'])
        elif requirement == 'exact policy language':
            return '"' in response or 'policy states' in response_lower or 'according to' in response_lower
        elif requirement == 'specific amounts':
            return bool(re.search(r'\$[\d,]+|\d+%|\d+ (dollars?|percent)', response))
        elif requirement == 'currencies or percentages':
            return bool(re.search(r'\$|%|\d+\s*(dollars?|cents?|percent)', response))
        elif requirement == 'step-by-step process':
            return bool(re.search(r'(step \d|first|second|third|then|next|finally)', response_lower))
        elif requirement == 'time requirements':
            return bool(re.search(r'\d+\s*(days?|months?|years?|hours?)|within|deadline|due', response_lower))
        elif requirement == 'direct answer':
            return len(response.strip()) > 20
        elif requirement == 'policy reference':
            return any(term in response_lower for term in ['policy', 'document', 'according to', 'states'])
        
        return True
    
    def _addresses_question_word(self, question_word: str, response: str) -> bool:
        """Check if response addresses specific question words"""
        response_lower = response.lower()
        
        if question_word == 'what':
            return len(response) > 20  # Any substantial response
        elif question_word == 'when':
            return bool(re.search(r'\d+\s*(days?|months?|years?)|date|time|period|effective', response_lower))
        elif question_word == 'where':
            return any(term in response_lower for term in ['location', 'office', 'website', 'address', 'contact'])
        elif question_word == 'why':
            return any(term in response_lower for term in ['because', 'reason', 'due to', 'in order to'])
        elif question_word == 'how':
            return bool(re.search(r'(step|process|procedure|by|through)', response_lower))
        elif question_word == 'who':
            return any(term in response_lower for term in ['person', 'customer', 'insured', 'beneficiary', 'you'])
        elif question_word == 'which':
            return any(term in response_lower for term in ['type', 'kind', 'specific', 'particular'])
        
        return False
    
    def _detect_hallucinations(self, response: str) -> float:
        """Detect potential hallucinations in response"""
        hallucination_score = 0.0
        response_lower = response.lower()
        
        # Check for hallucination indicators
        for pattern in self.hallucination_indicators:
            if re.search(pattern, response_lower):
                hallucination_score += 0.2
        
        # Check for vague references
        vague_patterns = [
            r'this type of policy', r'policies like this', r'standard coverage',
            r'most insurance', r'typical benefits', r'common exclusions'
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, response_lower):
                hallucination_score += 0.15
        
        return min(hallucination_score, 1.0)
    
    def _check_terminology_consistency(self, response: str, context: str) -> float:
        """Check if response uses consistent terminology with context"""
        if not context:
            return 0.8  # Neutral score if no context
        
        # Extract key terms from both
        response_terms = set(re.findall(r'\b[A-Za-z]{4,}\b', response.lower()))
        context_terms = set(re.findall(r'\b[A-Za-z]{4,}\b', context.lower()))
        
        if not response_terms:
            return 0.8
        
        # Check how many response terms appear in context
        matching_terms = response_terms & context_terms
        consistency_score = len(matching_terms) / len(response_terms)
        
        return consistency_score
    
    def _calculate_confidence(self, factual: float, completeness: float, 
                            non_hallucination: float, terminology: float) -> float:
        """Calculate overall confidence score"""
        weights = {
            'factual': 0.4,
            'completeness': 0.3,
            'non_hallucination': 0.2,
            'terminology': 0.1
        }
        
        confidence = (
            weights['factual'] * factual +
            weights['completeness'] * completeness +
            weights['non_hallucination'] * non_hallucination +
            weights['terminology'] * terminology
        )
        
        return confidence

# Global validator instance
response_validator = ResponseValidator()

async def validate_response(question: str, response: str, context: str) -> ValidationResult:
    """Validate a response for accuracy and completeness"""
    return await response_validator.validate_response(question, response, context)

def get_response_confidence(question: str, response: str, context: str) -> float:
    """Get quick confidence score for a response"""
    import asyncio
    loop = asyncio.get_event_loop()
    
    if loop.is_running():
        # Create a new task if we're already in an event loop
        task = asyncio.create_task(validate_response(question, response, context))
        # This is a simplified sync version for quick scoring
        return 0.8  # Default confidence
    else:
        result = loop.run_until_complete(validate_response(question, response, context))
        return result.confidence_score
