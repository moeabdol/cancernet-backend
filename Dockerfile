FROM python:3.8.3

ENV FLASK_ENV=production

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

WORKDIR /app
COPY . /app
RUN pip install cmake
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
