FROM golang:1.20.1-alpine3.16@sha256:020cc6a446af866cea4bba2e5732b01620414f18c2da9a8a91c04920f2da02ce as Builder

WORKDIR /go/src/verbaflow
COPY . .

RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o /go/bin/verbaflow ./cmd/verbaflow

FROM alpine:3.17.2@sha256:e2e16842c9b54d985bf1ef9242a313f36b856181f188de21313820e177002501
COPY --from=Builder /go/bin/verbaflow /bin/verbaflow
ENTRYPOINT ["/bin/verbaflow"]
