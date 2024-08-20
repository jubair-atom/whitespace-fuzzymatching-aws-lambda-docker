FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

COPY src/* ${LAMBDA_TASK_ROOT}

CMD ["main.handler"]