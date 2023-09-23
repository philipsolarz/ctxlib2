from pathlib import Path
import httpx
from docarray import DocList
from docarray.documents import TextDoc
from docarray.index import InMemoryExactNNIndex
from rich.logging import RichHandler
import logging
import timeit

# Set up logging configuration with RichHandler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

class InputModel(TextDoc):
    pass

class OutputModel(TextDoc):
    pass

class Client:
    def __init__(self, host: str = "http://0.0.0.0:8000"):
        self.host = host            

    async def generate(self, cxts: DocList[InputModel]) -> DocList[OutputModel]:
        start_time = timeit.default_timer()
        async with httpx.AsyncClient() as ac:
            try:
                logger.debug(f"Initiating request to {self.host}/embeddings/generate")
                response = await ac.post(f"{self.host}/embeddings/generate", data=cxts.to_json())
                response.raise_for_status()
                logger.debug(f"Received response in {timeit.default_timer() - start_time:.2f} seconds")
                return DocList[OutputModel].from_json(response.content.decode())
            except httpx.HTTPError as err:
                logger.error(f"HTTP error occurred: {err}")
            except Exception as err:
                logger.error(f"An error occurred: {err}")

async def main():
    ctxlib = Client()
    db = InMemoryExactNNIndex[OutputModel]()
    root_dir = Path('~/github/cpython/Lib').expanduser()
    py_files = root_dir.rglob('*.py')
    
    logger.debug(f"Commencing iteration through .py files in {root_dir}")
    total_start_time = timeit.default_timer()
    for py_file in py_files:
        loop_start_time = timeit.default_timer()
        try:
            logger.debug(f"Constructing InputModel for {py_file}")
            ctx = InputModel(url=str(py_file))
            ctx.text = ctx.url.load()
            logger.debug(f"Constructed InputModel for {ctx.url} in {timeit.default_timer() - loop_start_time:.2f} seconds")
            
            logger.debug(f"Transmitting InputModel to client for embedding generation")
            r = await ctxlib.generate(DocList[InputModel]([ctx]))
            logger.debug(f"Embedding: {r[0].embedding[:5]}")
            db.index(r)
            logger.debug(f"Indexed {py_file} in {timeit.default_timer() - loop_start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error processing {py_file}: {e}")
        finally:
            logger.debug(f"Total time for processing {py_file}: {timeit.default_timer() - loop_start_time:.2f} seconds")
    logger.debug(f"Total execution time: {timeit.default_timer() - total_start_time:.2f} seconds")
    while True:
        query = InputModel(text=input("Ask something Python related: "))
        r = await ctxlib.generate(DocList[InputModel]([query]))
        logger.debug(f"Query: {query.text}")
        # find similar documents
        matches, scores = db.find(r[0].embedding, search_field='embedding', limit=5)

        print(f'{matches=}')
        print(f'{matches.url=}')
        print(f'{scores=}')

# Run the asynchronous main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
