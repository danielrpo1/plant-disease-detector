# Imagen para AWS Lambda (container)
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar modelo ONNX + meta (build: copiar desde artifacts/)
COPY model.onnx ${LAMBDA_TASK_ROOT}/model.onnx
COPY model.meta.json ${LAMBDA_TASK_ROOT}/model.meta.json
COPY handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY labels.py ${LAMBDA_TASK_ROOT}/labels.py

CMD ["handler.lambda_handler"]
