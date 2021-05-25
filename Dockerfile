FROM rust:1.50 as build

WORKDIR /usr/src/ort_test
COPY . .

RUN cargo install --path .

FROM gcr.io/distroless/cc-debian10

COPY --from=build /usr/local/cargo/bin/ort_test /usr/local/bin/ort_test
COPY tokenizer.json .
COPY gpt2.onnx .
COPY libonnxruntime.so.1.6.0 .
ENV LD_LIBRARY_PATH=.

CMD ["ort_test"]

