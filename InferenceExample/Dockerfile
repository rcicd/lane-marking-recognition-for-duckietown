FROM dustynv/jetson-inference:r32.4.4

SHELL ["/bin/bash", "-c"]

ENV TZ=Europe/Nicosia

WORKDIR /app
COPY RoadMarkupsSegmentationNet.engine /app/
COPY main.cu /app/
COPY CMakeLists.txt /app/

ARG BUILD_DEBUG=OFF

RUN mkdir build
WORKDIR /app/build
RUN cmake .. -DBUILD_DEBUG=${BUILD_DEBUG}
RUN make
RUN cp /app/build/nn /app/nn
WORKDIR /app
