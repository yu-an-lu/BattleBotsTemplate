FROM python:3

# Install dependencies
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --requirement /tmp/requirements.txt

# Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
