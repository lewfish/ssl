FROM vissl:1.0-cu102

COPY ./ssl/ /opt/src/ssl/
ENV PYTHONPATH=/opt/src/:$PYTHONPATH
WORKDIR /opt/src

CMD ["bash"]
