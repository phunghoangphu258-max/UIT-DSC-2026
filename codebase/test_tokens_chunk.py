def main() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")
    sentence = "Thời hạn cấp đăng ký xe máy là bao lâu?"

    token_ids = model.tokenizer(sentence, truncation=False)["input_ids"]

    print("Token count:", len(token_ids))
    print("Tokens:", model.tokenizer.convert_ids_to_tokens(token_ids))


if __name__ == "__main__":
    main()
