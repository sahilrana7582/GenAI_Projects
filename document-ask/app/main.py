from dotenv import load_dotenv

load_dotenv()

from core.llm import model


response = model.invoke("Hello")

print(response)
print(response.content)
