FROM python:3.11.3-bullseye as build

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/app

ENV HNSWLIB_NO_NATIVE=1
RUN pip install --upgrade pip

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM build

# ENV DATA_DIRECTORY=/tmp
# ENV CHROMA_PERSIST_DIRECTORY=./.chroma

# EXPOSE 5000

# Run the application:
COPY .database .database
COPY app.py .
CMD ["python", "app.py"]
