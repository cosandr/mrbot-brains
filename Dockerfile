ARG PY_VER=3.8
FROM python:${PY_VER}-bullseye AS builder

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && \
    apt-get upgrade -y && \
    echo 'apt: install deps' && \
    apt-get -qq install build-essential git pkg-config libhdf5-103 libhdf5-dev

ENV PATH="/opt/venv/bin:$PATH" \
    PIP_ARGS="-i https://www.dresrv.com/pip --extra-index-url https://pypi.org/simple"

COPY requirements.txt /tmp/req.txt

RUN python -m venv /opt/venv && \
    pip install -U pip wheel setuptools && \
    echo 'pip: install deps' && \
    sed -i 's/^opencv.*//g' /tmp/req.txt && \
    pip install ${PIP_ARGS} -r /tmp/req.txt && \
    pip install -i https://www.dresrv.com/pip \
        opencv-python-headless

FROM python:${PY_VER}-slim-bullseye

ARG OVERLAY_VERSION="v2.2.0.3"

ADD https://github.com/just-containers/s6-overlay/releases/download/${OVERLAY_VERSION}/s6-overlay-amd64.tar.gz /tmp/
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    LANGUAGE="en_US.UTF-8" \
    LANG="en_US.UTF-8" \
    TERM="xterm" \
    PYTHONUNBUFFERED="1"

RUN apt-get update && \
    apt-get upgrade -y && \
    echo '**** apt: install deps ****' && \
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        sudo locales ffmpeg tar libilmbase-dev libopenexr-dev && \
    echo '**** S6: Install ****' && \
    tar xzf /tmp/s6-overlay-amd64.tar.gz -C / && \
    echo '**** user: Create abc user and group ****' && \
    groupadd --gid 1000 abc && useradd --create-home --gid 1000 --uid 1000 abc && \
    echo '**** locale setup ****' && \
    locale-gen en_US.UTF-8 && \
    echo "**** cleanup ****" && \
    apt-get clean && \
    rm -rf \
        /tmp/* \
        /var/lib/apt/lists/* \
        /var/tmp/*

VOLUME [ "/app", "/data", "/upload" ]

COPY docker/root/ /
COPY . /app

ENTRYPOINT [ "/init" ]
