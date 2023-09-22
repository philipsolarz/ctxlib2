from pathlib import Path
import httpx
from docarray import DocList
from docarray.documents import TextDoc

class InputModel(TextDoc):
    pass

async def main():
    # Get the list of all text files in the project directory and its subdirectories
    text_files = [file for file in Path('.').rglob('*.txt')]
    
    # Load the content of each file and create a list of InputModel instances
    input_models: List[InputModel] = []
    for text_file in text_files:
        try:
            # Create InputModel instance
            input_model = InputModel(url=str(text_file))
            # Load the text content using the .load method and set it to input_model.text
            with text_file.open() as file:
                input_model.text = file.read()
            input_models.append(input_model)
        except Exception as e:
            print(f"Error reading {text_file}: {e}")
    
    # Prepare the payload as a DocList
    payload = DocList[InputModel](input_models)
    
    # Send a POST request to the API to generate embeddings
    async with httpx.AsyncClient() as ac:
        try:
            response = await ac.post("http://0.0.0.0:8000/embeddings/generate", data=payload.to_json())
            response.raise_for_status()
            
            # Parse the response back into a DocList
            output_docs = DocList[TextDoc].from_json(response.content.decode())
            print(output_docs[0].embedding)
        except httpx.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

# Run the asynchronous main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
