import os
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field
import config

# 1. Define the Data Schema
class CustomerData(BaseModel):
    """
    Schema for the input features. 
    Use Field aliases to match CSV column names with spaces.
    """
    age: int = Field(alias="Age")
    job: int = Field(alias="Job")
    housing: str = Field(alias="Housing")
    saving_accounts: Optional[str] = Field("unknown", alias="Saving accounts")
    checking_account: Optional[str] = Field("unknown", alias="Checking account")
    credit_amount: int = Field(alias="Credit amount")
    duration: int = Field(alias="Duration")
    purpose: str = Field(alias="Purpose")
    risk: str = Field(alias="Risk")

    class Config:
        # This allows the model to be populated using the field names OR the aliases
        populate_by_name = True

class RiskExplanation(BaseModel):
    """Schema for validating the LLM output."""
    description: str

# 2. Define the LLM Engine Class
class CreditLLMEngine:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None):
        """
        The code logic stays here, but the values come from config.py.
        Priority: Argument > Environment Variable > Config File
        """
        # Resolve values dynamically
        self.model = model or os.getenv("MODEL_NAME") or config.LLM_MODEL
        final_url = base_url or os.getenv("BASE_URL") or config.BASE_URL
        final_key = api_key or os.getenv("API_KEY") or config.API_KEY

        self.client = OpenAI(
            base_url=final_url,
            api_key=final_key
        )

    def _generate_prompt(self, data: CustomerData) -> str:
        return f"""
        SYSTEM: You are a world-class financial risk communicator. 
        Your goal is to translate complex credit data into friendly, actionable advice.

        DATA INPUT:
        - Risk Classification: {data.risk}
        - Loan Details: {data.credit_amount} USD over {data.duration} months.
        - Purpose: {data.purpose}
        - Financial Profile: {data.checking_account} checking balance, {data.saving_accounts} savings.
        - Personal: Age {data.age}, Job Level {data.job}, Housing: {data.housing}.

        TASK:
        1. ANALYZE (Internal): Briefly identify the top 2 factors contributing to the {data.risk} rating.
        2. EXPLAIN: Write a 3-5 line explanation for the customer. 
           - Use the "Friend at a Coffee Shop" tone.
           - Avoid jargon like "debt-to-income" or "liquidity."
           - Focus on what the data means for their life.
        3. ADVISE: If the risk is 'bad', provide 1 specific, non-obvious tip to improve. 

        CONSTRAINTS:
        - No bullet points.
        - No bold text.
        - Start directly with the explanation.
        """

    def get_description(self, data: CustomerData) -> str:
        """Processes the CustomerData and returns only the text description."""
        try:
            prompt = self._generate_prompt(data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional financial advisor who speaks simply. "
                                   "You never use technical jargon or bullet points."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.6 
            )
            
            content = response.choices[0].message.content
            
            # Validation via Pydantic
            result = RiskExplanation(description=content)
            return result.description

        except Exception as e:
            # Better error logging for debugging during your M.Tech work
            print(f"DEBUG: LLM Inference failed: {e}") 
            return f"I'm sorry, I couldn't generate an explanation right now. (Error: {str(e)})"

# --- Usage Example (How to call this in your main app) ---
# if __name__ == "__main__":
#     # Initialize (will use env vars or defaults automatically)
#     engine = CreditLLMEngine()
#
#     # Example dictionary (mocking a row from your df)
#     example_row = {
#         "Age": 33, "Job": 2, "Housing": "own", 
#         "Saving accounts": "little", "Checking account": "moderate",
#         "Credit amount": 1169, "Duration": 6, "Purpose": "radio/TV", "Risk": "good"
#     }
#
#     # Create Pydantic model from dict
#     customer = CustomerData(**example_row)
#
#     # Get results
#     text = engine.get_description(customer)
#     print(text)