import os
from dotenv import load_dotenv

from langchain.llms import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    print(OPENAI_API_KEY)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    result = llm.predict("Give me a series or 5 to 7 poses to sequence in a yoga flow focused on the heart chakra.")
    print(result)

if __name__ == "__main__":
    main()
