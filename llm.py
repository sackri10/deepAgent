# agent/llm.py

from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# For Anthropic Claude 3.5 Sonnet: "anthropic.claude-3-5-sonnet-20240620-v1:0"
# Ensure your AWS credentials are configured (e.g., via environment variables)
# llm = ChatBedrockConverse(
#     model="anthropic.claude-3-5-sonnet-20240620-v1:0",
#     region_name="us-east-1",  # or your preferred region
#     # credentials_profile_name="your-aws-profile" # if needed
# )
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
# llm= ChatOpenAI(
#     model="gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))

llm_aws = ChatBedrockConverse(
    model="apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
    temperature=0,
    max_tokens=None,
    region_name=os.getenv("AWS_BEARER_TOKEN_BEDROCK_REGION"),
    endpoint_url="https://bedrock-runtime.ap-south-1.amazonaws.com",
    credentials_profile_name=None,  # Use default AWS credentials
    # other params...
)

model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # exact model ID
        temperature=0,
    )
sub_agent_model =ChatAnthropic(
        model="claude-haiku-4-5",  # exact model ID
        temperature=0,
    )