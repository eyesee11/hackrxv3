import google.generativeai as genai
import together
import asyncio
import time
from typing import Optional
from cache_system import cache, CircuitBreaker
import os
from dotenv import load_dotenv

load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
together.api_key = os.getenv("TOGETHER_API_KEY")

# Circuit breakers for each service
gemini_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
together_breaker = CircuitBreaker(failure_threshold=3, timeout=60)

class LLMService:
    """Optimized LLM service with multiple providers and fallbacks"""
    
    def __init__(self):
        # Gemini 2.0 Flash Lite configuration for maximum speed
        self.gemini_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 200,  # Keep responses concise for speed
            "response_mime_type": "text/plain",
        }
        
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite",
            generation_config=self.gemini_config,
        )
        
        # Together.ai configuration as fallback
        self.together_config = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "max_tokens": 150,
            "temperature": 0.1,
            "stop": ["Q:", "\n\n"]
        }
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate response with primary and fallback LLMs"""
        
        # Create optimized prompt
        prompt = self._create_optimized_prompt(query, context)
        
        try:
            # Try Gemini first (primary)
            response = await gemini_breaker.call(
                self._generate_with_gemini, 
                prompt
            )
            return response.strip()
            
        except Exception as e:
            print(f"❌ Gemini error: {str(e)}, falling back to Together.ai")
            
            try:
                # Fallback to Together.ai
                response = await together_breaker.call(
                    self._generate_with_together, 
                    query, 
                    context
                )
                return response.strip()
                
            except Exception as e2:
                print(f"❌ Together.ai error: {str(e2)}")
                return "I apologize, but I'm unable to generate a response at the moment due to service issues."
    
    def _create_optimized_prompt(self, query: str, context: str) -> str:
        """Create enhanced prompt with few-shot examples and chain-of-thought reasoning"""
        
        # Determine query type for specialized prompting
        query_type = self._classify_query_type(query)
        
        if query_type == "coverage":
            return self._create_coverage_prompt(query, context)
        elif query_type == "exclusions":
            return self._create_exclusions_prompt(query, context)
        elif query_type == "claims":
            return self._create_claims_prompt(query, context)
        elif query_type == "amounts":
            return self._create_amounts_prompt(query, context)
        else:
            return self._create_general_prompt(query, context)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for specialized prompting"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cover', 'included', 'benefit', 'protection']):
            return "coverage"
        elif any(word in query_lower for word in ['exclude', 'not cover', 'exception', 'limitation']):
            return "exclusions"
        elif any(word in query_lower for word in ['claim', 'file', 'procedure', 'process', 'how to']):
            return "claims"
        elif any(word in query_lower for word in ['cost', 'premium', 'deductible', 'amount', 'limit', 'maximum']):
            return "amounts"
        else:
            return "general"
    
    def _create_coverage_prompt(self, query: str, context: str) -> str:
        """Specialized prompt for coverage questions"""
        return f"""You are an expert insurance policy analyst specializing in coverage analysis. Answer coverage questions with precision and clarity.

Context from Policy Document:
{context}

Question: {query}

Example Analysis:
Q: "What does this policy cover for dental procedures?"
A: "Based on the policy document, dental coverage includes: [specific items listed]. The policy covers up to [amount/percentage] for [specific procedures]. Coverage applies when [specific conditions]."

Q: "Is emergency care covered?"
A: "Yes, emergency care is covered. The policy states: [exact quote]. Coverage includes [specific details] with [any limitations or conditions]."

Analysis Steps:
1. Identify what specific coverage the question asks about
2. Find exact policy language related to that coverage
3. Note any amounts, percentages, or conditions
4. Include any relevant limitations or requirements

Your Answer (be specific and quote policy language):"""

    def _create_exclusions_prompt(self, query: str, context: str) -> str:
        """Specialized prompt for exclusions questions"""
        return f"""You are an expert insurance policy analyst specializing in exclusions and limitations. Provide clear explanations of what is NOT covered.

Context from Policy Document:
{context}

Question: {query}

Example Analysis:
Q: "What is not covered under this policy?"
A: "The policy specifically excludes: [list exact exclusions from policy]. Additionally, limitations include: [specific limitations]. These exclusions mean [practical implications]."

Q: "Are pre-existing conditions excluded?"
A: "According to the policy: [exact policy language]. This applies to [specific scenarios] and means [practical explanation]."

Analysis Steps:
1. Identify the specific exclusion or limitation being asked about
2. Find exact exclusion language in the policy
3. Explain the practical implications
4. Note any exceptions to the exclusion

Your Answer (quote exact exclusion language):"""

    def _create_claims_prompt(self, query: str, context: str) -> str:
        """Specialized prompt for claims procedure questions"""
        return f"""You are an expert insurance policy analyst specializing in claims procedures. Provide step-by-step guidance based on policy requirements.

Context from Policy Document:
{context}

Question: {query}

Example Analysis:
Q: "How do I file a claim?"
A: "To file a claim under this policy: Step 1: [specific requirement]. Step 2: [next requirement]. Step 3: [final step]. You must do this within [time limit] of [triggering event]. Required documentation includes: [specific documents]."

Q: "What's the claims process for emergency situations?"
A: "For emergency claims: [immediate steps required]. Within [timeframe]: [additional requirements]. The policy states: [exact quote about emergency procedures]."

Analysis Steps:
1. Identify the specific type of claim being asked about
2. Find step-by-step procedures in the policy
3. Note all time requirements and deadlines
4. List required documentation
5. Include any special procedures for different claim types

Your Answer (provide step-by-step instructions):"""

    def _create_amounts_prompt(self, query: str, context: str) -> str:
        """Specialized prompt for financial/amount questions"""
        return f"""You are an expert insurance policy analyst specializing in financial terms, premiums, deductibles, and coverage limits. Provide precise numerical information.

Context from Policy Document:
{context}

Question: {query}

Example Analysis:
Q: "What is the deductible for this policy?"
A: "The deductible is [exact amount] for [specific coverage type]. This means you pay the first [amount] of covered expenses before the policy begins paying. For [different coverage types], the deductible is [different amounts if applicable]."

Q: "What are the coverage limits?"
A: "Coverage limits are: [specific amounts for each type]. Annual maximum: [amount]. Per-incident maximum: [amount]. These limits apply to [specific explanation of what's counted toward limits]."

Analysis Steps:
1. Identify the specific financial aspect being asked about
2. Find exact amounts, percentages, or formulas in the policy
3. Explain how the amounts are calculated or applied
4. Note any variations for different situations
5. Include any additional fees or charges

Your Answer (include all specific amounts and calculations):"""

    def _create_general_prompt(self, query: str, context: str) -> str:
        """General prompt for other types of questions"""
        return f"""You are an expert insurance policy analyst. Based on the provided policy document context, answer the question accurately and comprehensively.

Context from Policy Document:
{context}

Question: {query}

Example Analysis:
Q: "When does this policy take effect?"
A: "According to the policy document: [exact quote]. This means [practical explanation]. The policy becomes active [specific timing] and coverage begins [when coverage starts]."

Analysis Steps:
1. Understand exactly what the question is asking
2. Find relevant information in the policy document
3. Quote exact policy language when possible
4. Provide clear, practical explanation
5. Note any important conditions or limitations

Instructions:
- Provide a direct, factual answer based only on the policy document
- Quote exact policy language when relevant
- Include specific details like time periods, percentages, amounts
- If information is not in the context, state "Information not available in the provided document"
- Explain the practical implications of policy terms
- Use the exact terminology from the policy document

Your Answer:"""
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini 2.0 Flash Lite"""
        response = await self.gemini_model.generate_content_async(prompt)
        return response.text
    
    async def _generate_with_together(self, query: str, context: str) -> str:
        """Generate response using Together.ai as fallback"""
        prompt = f"Context: {context}\n\nQ: {query}\nA:"
        
        response = together.Complete.create(
            prompt=prompt,
            **self.together_config
        )
        
        return response['output']['choices'][0]['text']
    
    async def generate_answer_with_context(self, question: str, context: str) -> str:
        """Main interface for generating answers with context"""
        
        # Check cache first
        cache_key = f"{question}||{context[:100]}"  # Use first 100 chars of context for key
        cached_result = cache.get_cached_query(cache_key)
        
        if cached_result and 'answer' in cached_result:
            return cached_result['answer']
        
        # Generate new answer
        start_time = time.time()
        answer = await self.generate_response(question, context)
        response_time = (time.time() - start_time) * 1000
        
        # Cache the result
        cache.cache_query_result(cache_key, {
            'answer': answer,
            'response_time_ms': response_time
        })
        
        return answer

# Global LLM service instance
llm_service = LLMService()

# Convenience functions
async def generate_response(query: str, context: str) -> str:
    """Generate response with context"""
    return await llm_service.generate_response(query, context)

async def generate_answer_with_context(question: str, context: str) -> str:
    """Generate answer with context and caching"""
    return await llm_service.generate_answer_with_context(question, context)
