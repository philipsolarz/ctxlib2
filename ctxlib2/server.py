from fastapi import FastAPI, HTTPException, Depends
from docarray import BaseDoc, DocList
from docarray.documents import TextDoc
from docarray.typing import AnyEmbedding
from docarray.base_doc import DocArrayResponse
from sentence_transformers import SentenceTransformer
from typing import List

api = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

class InputModel(TextDoc):
    pass

class OutputModel(TextDoc):
    pass

@api.post("/embeddings/generate", response_model=List[OutputModel], response_class=DocArrayResponse)
def generate_embeddings(cxts: List[InputModel]) -> List[OutputModel]:
    # cast cxts from List[InputModel] to DocList[InputModel]
    cxts = DocList[InputModel](cxts)
    for ctx in cxts:
        try:
            # if ctx.text is None:
            #     ctx.text = ctx.url.load()
            ctx.embedding = model.encode(ctx.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return list(cxts)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:api", host="0.0.0.0", workers=1, reload=True)

"""
1. Make model customizable, and fetched from a central local place instead of downloading, or at least look for local model first before downloading.
2. How utilize CUDA on Arcus?
"""