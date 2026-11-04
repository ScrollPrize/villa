#pragma once
// Deterministic JSON encoding of typed transaction requests and receipts,
// shared by C++, Python and the hosted wire protocol.
//
// Binary operation payloads (geometry arrays, edits, rectangles) are not
// embedded: the JSON names each by its SHA-256 and length, and the bytes
// travel separately (`payloads`, keyed by that hash). The request digest is the
// SHA-256 of the canonical JSON without the request UUID, so it covers the
// command version, store/generation, author, preconditions, ordered
// operations and payload identities, but not how or when bytes were staged.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vc_object_store/objects/Transactions.hpp"

namespace vc::objects {

using TransactionPayloads = std::map<std::string, std::vector<uint8_t>>;

// Canonical JSON text; when `payloads` is given it receives every payload.
std::string transactionRequestToJson(const TransactionRequest&, TransactionPayloads* payloads = nullptr);
// Every payload named by the JSON must be supplied and match its hash/length.
TransactionRequest transactionRequestFromJson(const std::string& json, const TransactionPayloads& payloads);
std::string transactionRequestDigest(const TransactionRequest&);
// Hash of the payload set a request needs, without materializing the JSON's
// bytes: lets a client re-stage exactly these after an upload expires.
std::vector<std::string> transactionPayloadHashes(const TransactionRequest&);
std::string payloadHash(const std::vector<uint8_t>&);

// Record encodings used by CreateObjectOp/UpdateObjectOp and AddEdgeOp/UpdateEdgeOp.
std::string objectRecordJson(const ObjectRecord&);
std::string edgeRecordJson(const EdgeRecord&);

std::string transactionReceiptToJson(const TransactionReceipt&);
TransactionReceipt transactionReceiptFromJson(const std::string&);
std::string storeStatusToJson(const StoreStatus&);
std::string changePageToJson(const ChangePage&);

} // namespace vc::objects
