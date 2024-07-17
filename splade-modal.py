from fastapi.responses import JSONResponse
from modal import Image, Mount, Volume, Secret, App, method, web_endpoint, enter, gpu
from pydantic import BaseModel

# This is copied from: https://github.com/pinecone-io/examples/blob/2f51ddfd12a08f2963cc2849661fab51afdeedc6/learn/search/semantic-search/sparse/splade/splade-vector-generation.ipynb#L10
# Which is recommended here: https://docs.pinecone.io/docs/hybrid-search

image = Image.debian_slim().pip_install("torch", "transformers")
volume = Volume.from_name("splade-model-cache-vol")

app = App()
# gpu = gpu(gpu.T4, count=1)

CACHE_DIR = "/cache"

class Body(BaseModel):
    text: str
    is_doc: bool


@app.cls(
    image=image,
#     cloud="gcp",
    cpu=2,
#     gpu="any",
#     memory=2048,
    keep_warm=3,
    container_idle_timeout=120,
    volumes={CACHE_DIR: volume},
    secrets=[Secret.from_dict({"TORCH_HOME": CACHE_DIR, "TRANSFORMERS_CACHE": CACHE_DIR})],
)
class SPLADE:
    @enter()
    def on_enter(self):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

#         model = "naver/splade-cocondenser-ensembledistil"
        # check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model)
#         self.model = AutoModelForMaskedLM.from_pretrained(model)

        doc_model_id = "naver/efficient-splade-VI-BT-large-doc"
        doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id)
        doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id)

        query_model_id = "naver/efficient-splade-VI-BT-large-query"
        query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
        query_model = AutoModelForMaskedLM.from_pretrained(query_model_id)

        self.doc_tokenizer = doc_tokenizer
        self.doc_model = doc_model
        self.query_tokenizer = query_tokenizer
        self.query_model = query_model

        # move to gpu if available
        self.doc_model.to(self.device)
        self.query_model.to(self.device)

    @web_endpoint(method="POST")
    def vector(self, body: Body):
        import torch
        from transformers.tokenization_utils_base import TruncationStrategy

        is_doc = body.is_doc
        text = body.text

        tokenizer = self.doc_tokenizer if is_doc else self.query_tokenizer
        model = self.doc_model if is_doc else self.query_model

        max_length = tokenizer.model_max_length
        inputs = tokenizer(
            text,
            truncation=TruncationStrategy.LONGEST_FIRST,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        inter = torch.log1p(torch.relu(logits[0]))
        token_max = torch.max(inter, dim=0)  # sum over input tokens
        nz_tokens = torch.where(token_max.values > 0)[0]
        nz_weights = token_max.values[nz_tokens]

        order = torch.sort(nz_weights, descending=True)
        nz_weights = nz_weights[order[1]]
        nz_tokens = nz_tokens[order[1]]

        response = {
            "indices": nz_tokens.cpu().numpy().tolist(),
            "values": nz_weights.cpu().numpy().tolist(),
        }

        return JSONResponse(content=response)
