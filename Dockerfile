FROM huggingface/transformers-pytorch-cpu:latest

WORKDIR /app

COPY requirements.txt .
COPY ./app /app/app


RUN pip install -r requirements.txt

CMD ["python3", "app/run.py"]


