FROM yeop2/defect-detection as build

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY --from=build /app /app
WORKDIR /app

COPY . . 

RUN apt update
RUN apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev
    
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "main.py"]
