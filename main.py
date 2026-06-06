from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

import filter_app
import nsfw_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(filter_app.lifespan(app))
        await stack.enter_async_context(nsfw_app.lifespan(app))
        yield


app = FastAPI(title="Gwangsan AI", lifespan=lifespan)


@app.get("/health")
def health():
    ready = (
        filter_app.model is not None
        and filter_app.tokenizer is not None
        and nsfw_app.model is not None
        and nsfw_app.processor is not None
    )
    return {"status": "ok" if ready else "loading"}


@app.post("/predict")
def predict(request: filter_app.TextRequest):
    return filter_app.predict(request)


@app.post("/nsfw")
async def nsfw(file: nsfw_app.UploadFile = nsfw_app.File(...)):
    return await nsfw_app.predict(file)
