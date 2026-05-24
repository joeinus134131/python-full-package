import os
from openai import OpenAI

ai_api_key = "ccc"

client = OpenAI(
  api_key=ai_api_key,
  base_url='https://ai-gateway.vercel.sh/v1'
)

response = client.chat.completions.create(
  model='openai/gpt-5.5',
  messages=[
    {
      'role': 'user',
      'content': 'Why is the sky blue?'
    }
  ]
)