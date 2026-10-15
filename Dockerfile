FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/.fastembed \
    RETRIEVAL_MODE=hybrid LLM_PROVIDER=openai_compat LLM_MODEL=openai/gpt-oss-20b
WORKDIR /home/user/app
COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
COPY --chown=user . .
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
EXPOSE 10000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
